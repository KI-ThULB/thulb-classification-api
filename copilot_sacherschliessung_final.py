import requests
import xml.etree.ElementTree as ET
import re
import logging
import json
import aiohttp
import asyncio
import urllib.parse
import ssl
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pydantic_settings import BaseSettings
from pydantic import HttpUrl
from datetime import datetime, timedelta

class Settings(BaseSettings):
    """Konfigurationseinstellungen für die Anwendung."""
    DNB_SRU_URL: HttpUrl = "https://services.dnb.de/sru/dnb"
    RVK_API_URL: HttpUrl = "https://rvk.uni-regensburg.de/api"
    RETRY_ATTEMPTS: int = 5
    RETRY_DELAY: float = 2.0
    CACHE_TTL: int = 3600
    LOG_LEVEL: str = "INFO"
    VERIFY_SSL: bool = True
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False

@dataclass
class NotationMapping:
    rvk: str
    thulb: str
    description: str = ""
    
    def __post_init__(self):
        if not self.rvk or not self.thulb:
            raise ValueError("RVK and ThULB notations must not be empty")

@dataclass
class RVKNode:
    notation: str
    benennung: str
    has_children: bool
    register: List[str]
    score: float = 0.0

    def __post_init__(self):
        if not self.notation:
            raise ValueError("Notation must not be empty")

@dataclass
class BookMetadata:
    idn: str
    titel: str
    authors: List[str]
    publication_year: Optional[str]
    subjects: List[str]
    classifications: List[str]
    rvk_notations: List[Dict[str, str]] = field(default_factory=list)
    thulb_notations: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.idn or not self.titel:
            raise ValueError("IDN and title must not be empty")
    
    def to_dict(self) -> dict:
        return {
            'idn': self.idn,
            'titel': self.titel,
            'authors': self.authors,
            'publication_year': self.publication_year,
            'subjects': self.subjects,
            'classifications': self.classifications,
            'rvk_notations': self.rvk_notations,
            'thulb_notations': self.thulb_notations
        }

class Cache:
    """Simple in-memory cache with TTL."""
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = (datetime.now(), value)

    def clear(self):
        self._cache.clear()

class RVKClient:
    def __init__(self, settings: Settings):
        self.base_url = str(settings.RVK_API_URL)
        self.session = None
        self.cache = Cache(settings.CACHE_TTL)
        self.settings = settings
        self.ssl_context = None if settings.VERIFY_SSL else self._create_insecure_ssl_context()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _create_insecure_ssl_context():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_nodes(self, term: str) -> List[RVKNode]:
        """Sucht nach RVK-Notationen mit Cache-Unterstützung."""
        if not term or len(term.strip()) < 3:
            self.logger.warning(f"Search term too short or empty: '{term}'")
            return []

        cache_key = f"rvk_search_{term.lower()}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            nodes = await self._search_with_retry(term)
            if nodes:
                self.cache.set(cache_key, nodes)
            return nodes
        except Exception as e:
            self.logger.error(f"RVK search error for term '{term}': {e}", exc_info=True)
            return []

    async def _search_with_retry(self, term: str) -> List[RVKNode]:
        delay = self.settings.RETRY_DELAY
        for attempt in range(self.settings.RETRY_ATTEMPTS):
            try:
                return await self._search_direct(term)
            except aiohttp.ClientError as e:
                if attempt == self.settings.RETRY_ATTEMPTS - 1:
                    raise
                self.logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(delay)
                delay *= 2
        return []

    async def _search_direct(self, term: str) -> List[RVKNode]:
        """Sucht direkt nach RVK-Notationen mit verbessertem Logging."""
        url = f"{self.base_url}/json/nodes/{urllib.parse.quote(term)}"
        
        if not self.session:
            raise RuntimeError("Client session not initialized")

        self.logger.debug(f"Searching RVK API with URL: {url}")
        
        async with self.session.get(url) as response:
            response.raise_for_status()
            self.logger.debug(f"RVK API response status: {response.status}")
            
            try:
                data = await response.json()
                self.logger.debug(f"RVK API response data: {data}")
                
                if not data:
                    self.logger.debug(f"No data found for term: {term}")
                    return []

                nodes = []
                if 'node' in data:
                    node_data = data['node']
                    if isinstance(node_data, list):
                        nodes.extend(self._create_node(node) for node in node_data)
                    else:
                        nodes.append(self._create_node(node_data))

                    self.logger.debug(f"Found {len(nodes)} nodes for term: {term}")
                    return nodes
                    
                self.logger.debug(f"No nodes found in response for term: {term}")
                return []
                
            except Exception as e:
                self.logger.error(f"Error processing RVK response for term {term}: {e}")
                raise

    @staticmethod
    def _create_node(node_data: Dict) -> RVKNode:
        return RVKNode(
            notation=node_data['notation'],
            benennung=node_data.get('benennung', ''),
            has_children=node_data.get('has_children', 'no') == 'yes',
            register=node_data.get('register', [])
        )
    
class RVKThULBMapper:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.mapping_rules: List[NotationMapping] = []
        self._load_mappings()
        self._mapping_cache: Dict[str, Optional[NotationMapping]] = {}

    def _load_mappings(self):
        try:
            if self.config_path and os.path.exists(self.config_path):
                self._load_from_file()
            else:
                self._load_default_mappings()
            self._validate_mappings()
        except Exception as e:
            self.logger.error(f"Error loading mappings: {e}", exc_info=True)
            raise

    def _load_from_file(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.mapping_rules = [
                NotationMapping(**mapping)
                for mapping in config.get('mappings', [])
            ]

    def _load_default_mappings(self):
        """Lädt die Standard-Mappings für medizinische Fachgebiete."""
        self.mapping_rules = [
            # Biochemie
            NotationMapping(r"WW 1500-1999", "MED GC", "Biochemie"),
            NotationMapping(r"WW 2000-2999", "MED GC", "Biochemie - Spezielle Gebiete"),
            NotationMapping(r"WW 3000-3999", "MED GC", "Biochemische Prozesse"),
            
            # Anatomie und Physiologie
            NotationMapping(r"WW 1000-1619", "MED G", "Morphologie, Anatomie und Physiologie allgemein"),
            NotationMapping(r"WW 1000-1459", "MED GA", "Morphologie und topographische Anatomie"),
            NotationMapping(r"WW 1450-1459", "MED GC", "Histologie"),
            NotationMapping(r"WW 1460-1539", "MED GC", "Physiologie"),
            NotationMapping(r"WW 1540-9900", "MED GH", "Physiologie spezielle Bereiche"),
            
            # Allgemeine Medizin und Zeitschriften
            NotationMapping(r"XB 0990-1999", "MED AA", "Medizinische Zeitschriften"),
            NotationMapping(r"XB 2000-6399", "MED AC", "Medizinische Nachschlagewerke"),
            NotationMapping(r"XC.*", "MED AK", "Geschichte der Medizin"),
            NotationMapping(r"XD 1000-1099", "MED KA", "Krankenhauswesen"),
            
            # Klinische Fächer
            NotationMapping(r"YB.*", "MED UI", "Innere Medizin"),
            NotationMapping(r"YC.*", "MED UI", "Kardiologie"),
            NotationMapping(r"YD.*", "MED UE", "Infektionskrankheiten"),
            NotationMapping(r"YE.*", "MED UD", "Tuberkulose"),
            
            # Neue Mappings für weitere Fachgebiete
            NotationMapping(r"XR.*", "MED UH", "Dermatologie und Venerologie"),
            NotationMapping(r"WE.*", "MED UH", "Haut- und Geschlechtskrankheiten"),
            NotationMapping(r"WH.*", "MED UG", "Gynäkologie und Geburtshilfe"),
            NotationMapping(r"WJ.*", "MED UN", "Kinderheilkunde"),
            NotationMapping(r"WL.*", "MED UK", "Psychiatrie, Neurologie"),
            NotationMapping(r"WM.*", "MED UK", "Psychiatrie"),
            NotationMapping(r"WN.*", "MED UK", "Neurologie"),
            
            # Geburtshilfe und Gynäkologie
            NotationMapping(r"WH 1000-9999", "MED UG", "Gynäkologie und Geburtshilfe"),
            NotationMapping(r"YQ 1000-9999", "MED UG", "Geburtshilfe"),
            NotationMapping(r"YQ.*", "MED UG", "Geburtshilfe und Frauenheilkunde"),
            
            # Pharmakologie
            NotationMapping(r"WB.*", "MED P", "Pharmakologie"),
            NotationMapping(r"XD 4000-6000", "MED P", "Arzneimittelkunde"),
            
             # Gynäkologie und Geburtshilfe
            NotationMapping(r"YQ.*", "MED UG", "Gynäkologie und Geburtshilfe"),
            NotationMapping(r"WH 6500-7000", "MED UG", "Schwangerschaft und Geburt"),
            NotationMapping(r"YQ 4000-5000", "MED UG", "Schwangerschaftsbetreuung"),

            # Allgemeine Bereiche
            NotationMapping(r"W.*", "MED A", "Medizin allgemein"),
            NotationMapping(r"X.*", "MED A", "Medizin allgemein"),
            NotationMapping(r"Y.*", "MED U", "Klinische Fächer"),
            
            # Spezielle Bereiche
            NotationMapping(r"XA.*", "MED AB", "Medizinische Bibliographien"),
            NotationMapping(r"XB 0990-1999", "MED AA", "Medizinische Zeitschriften"),
            NotationMapping(r"XB 2000-6399", "MED AC", "Medizinische Nachschlagewerke"),
            NotationMapping(r"XC.*", "MED AK", "Geschichte der Medizin"),
            NotationMapping(r"XD 1000-1099", "MED KA", "Krankenhauswesen"),
            NotationMapping(r"XD.*", "MED K", "Gesundheitswesen"),
            NotationMapping(r"XE.*", "MED L", "Medizinische Ausbildung"),
            NotationMapping(r"XF.*", "MED M", "Medizinische Forschung"),
            NotationMapping(r"XG.*", "MED N", "Medizinische Dokumentation"),
            NotationMapping(r"YF.*", "MED UF", "Stoffwechselkrankheiten"),
            NotationMapping(r"YG.*", "MED UG", "Urologie"),
            NotationMapping(r"YH.*", "MED UH", "Dermatologie"),
            NotationMapping(r"YI.*", "MED UI", "Allergologie"),
            NotationMapping(r"YK.*", "MED UK", "Neurologie"),
            NotationMapping(r"YL.*", "MED UL", "Chirurgie"),
            NotationMapping(r"YM.*", "MED UM", "Orthopädie"),
            NotationMapping(r"YN.*", "MED UN", "Augenheilkunde"),
            NotationMapping(r"YO.*", "MED UO", "HNO-Heilkunde"),
            NotationMapping(r"YP.*", "MED UP", "Zahnmedizin"),

            # Pharmakologie in Schwangerschaft/Geburtshilfe
            NotationMapping(r"WB 5400-5490", "MED P", "Pharmakologie"),
            NotationMapping(r"YQ 4000", "MED UG", "Schwangerschaft und Geburt"),
            NotationMapping(r"XD 4500", "MED P", "Arzneimittelkunde"),
            
            # Spezifische Notationen
            NotationMapping(r"YQ 4100", "MED UG", "Schwangerschaftsbetreuung"),
            NotationMapping(r"YQ 4200", "MED UG", "Geburtshilfe"),
            NotationMapping(r"WB 5450", "MED P", "Medikamentöse Therapie"),
            
            # Allgemeine Bereiche
            NotationMapping(r"WB.*", "MED P", "Pharmakologie"),
            NotationMapping(r"YQ.*", "MED UG", "Gynäkologie und Geburtshilfe"),
            NotationMapping(r"XD 4000-6000", "MED P", "Arzneimittelkunde")
        ]

    def _validate_mappings(self):
        valid_rules = []
        for rule in self.mapping_rules:
            try:
                self._validate_single_mapping(rule)
                valid_rules.append(rule)
            except ValueError as e:
                self.logger.warning(f"Invalid mapping rule skipped: {e}")
        
        self.mapping_rules = valid_rules
        if not self.mapping_rules:
            raise ValueError("No valid mapping rules found")

    def _validate_single_mapping(self, rule: NotationMapping):
        if not rule.rvk or not rule.thulb:
            raise ValueError(f"Invalid mapping: {rule}")
            
        if '-' in rule.rvk:
            start, end = rule.rvk.split('-')
            start_num = self._extract_number(start)
            end_num = self._extract_number(end)
            if not (start_num and end_num and start_num <= end_num):
                raise ValueError(f"Invalid range in mapping rule: {rule.rvk}")
        elif '.*' not in rule.rvk and not self._extract_number(rule.rvk):
            raise ValueError(f"Invalid notation format: {rule.rvk}")

    @lru_cache(maxsize=1000)
    def map_notation(self, rvk_notation: str) -> Optional[NotationMapping]:
        if not rvk_notation:
            return None
        
        rvk_notation = rvk_notation.strip()
        if rvk_notation in self._mapping_cache:
            return self._mapping_cache[rvk_notation]
        
        result = self._find_mapping(rvk_notation)
        self._mapping_cache[rvk_notation] = result
        return result

    def _find_mapping(self, rvk_notation: str) -> Optional[NotationMapping]:
        try:
            for rule in self.mapping_rules:
                if rule.rvk == rvk_notation:
                    return rule
            
            for rule in self.mapping_rules:
                if '-' in rule.rvk and self._is_notation_in_range(rvk_notation, rule):
                    return rule
            
            for rule in self.mapping_rules:
                if '.*' in rule.rvk:
                    pattern = rule.rvk.replace('.*', '.*')
                    if re.match(pattern, rvk_notation):
                        return rule
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding mapping for {rvk_notation}: {e}")
            return None

    def _is_notation_in_range(self, notation: str, rule: NotationMapping) -> bool:
        try:
            start, end = rule.rvk.split('-')
            notation_num = self._extract_number(notation)
            start_num = self._extract_number(start)
            end_num = self._extract_number(end)
            
            if all(x is not None for x in [notation_num, start_num, end_num]):
                notation_prefix = re.match(r'[A-Z]+', notation).group()
                start_prefix = re.match(r'[A-Z]+', start).group()
                
                if notation_prefix == start_prefix:
                    return start_num <= notation_num <= end_num
            return False
        except Exception as e:
            self.logger.error(f"Error checking range for {notation}: {e}")
            return False

    @staticmethod
    def _extract_number(notation: str) -> Optional[int]:
        try:
            match = re.search(r'\d+', notation)
            return int(match.group()) if match else None
        except Exception:
            return None

    def clear_cache(self):
        self._mapping_cache.clear()
        self.map_notation.cache_clear()

class DNBProcessor:
    def __init__(self, settings: Settings, rvk_client: RVKClient, mapper: RVKThULBMapper):
        self.settings = settings
        self.rvk_client = rvk_client
        self.mapper = mapper
        self.cache = Cache(settings.CACHE_TTL)
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.namespaces = {
            'zs': 'http://www.loc.gov/zing/srw/',
            'marc': 'http://www.loc.gov/MARC21/slim'
        }
        self.marc_fields = {
            'title': ('245', 'a'),
            'subtitle': ('245', 'b'),
            'authors': [('100', 'a'), ('700', 'a')],
            'year': ('264', 'c'),
            'subjects': [('600', 'a'), ('610', 'a'), ('611', 'a'), 
                        ('630', 'a'), ('650', 'a'), ('651', 'a')],
            'classifications': ('082', 'a')
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                ssl=None if self.settings.VERIFY_SSL else ssl.create_default_context()
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def process_isbn_batch(self, isbns: List[str], batch_size: int = 5) -> Dict[str, Optional[BookMetadata]]:
        if not isbns:
            return {}

        results = {}
        valid_isbns = [isbn for isbn in isbns if self._validate_isbn(isbn)]
        
        for i in range(0, len(valid_isbns), batch_size):
            batch = valid_isbns[i:i + batch_size]
            batch_tasks = [self.process_isbn(isbn) for isbn in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for isbn, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing ISBN {isbn}: {result}")
                    results[isbn] = None
                else:
                    results[isbn] = result

        return results

    async def process_isbn(self, isbn: str) -> Optional[BookMetadata]:
        if not self._validate_isbn(isbn):
            self.logger.error(f"Invalid ISBN format: {isbn}")
            return None

        cache_key = f"book_metadata_{isbn}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.debug(f"Cache hit for ISBN {isbn}")
            return cached_result

        try:
            xml_data = await self._query_dnb_with_retry(isbn)
            if not xml_data:
                return None

            metadata = await self._process_metadata(xml_data)
            if metadata:
                self.cache.set(cache_key, metadata)
                return metadata

        except Exception as e:
            self.logger.error(f"Error processing ISBN {isbn}: {e}", exc_info=True)
            return None

    @staticmethod
    def _validate_isbn(isbn: str) -> bool:
        """Validiert das ISBN-Format mit erweiterter Prüfung."""
        if not isbn:
            return False
            
        # Entferne Bindestriche und Leerzeichen
        isbn = isbn.replace('-', '').replace(' ', '')
        
        # Akzeptiere sowohl ISBN-10 als auch ISBN-13
        if len(isbn) not in [10, 13]:
            return False
            
        if not isbn.isdigit():
            # Erlaube 'X' als letzte Ziffer bei ISBN-10
            if len(isbn) == 10 and isbn[:-1].isdigit() and isbn[-1] in ['X', 'x']:
                pass
            else:
                return False
        
        # ISBN-13 Prüfsumme
        if len(isbn) == 13:
            check_sum = sum((3 if i % 2 else 1) * int(d) for i, d in enumerate(isbn[:-1]))
            check_digit = (10 - (check_sum % 10)) % 10
            return check_digit == int(isbn[-1])
            
        # ISBN-10 Prüfsumme
        elif len(isbn) == 10:
            check_sum = sum((10 - i) * (int(d) if d.isdigit() else 10) 
                          for i, d in enumerate(isbn))
            return check_sum % 11 == 0
            
        return False
            
        check_sum = sum((3 if i % 2 else 1) * int(d) for i, d in enumerate(isbn[:-1]))
        check_digit = (10 - (check_sum % 10)) % 10
        
        return check_digit == int(isbn[-1])

    async def _query_dnb_with_retry(self, isbn: str) -> Optional[str]:
        """Verbesserte DNB-Abfrage mit alternativen Suchstrategien."""
        # Erste Suche mit voller ISBN
        params = {
            'version': '1.1',
            'operation': 'searchRetrieve',
            'query': f'num={isbn}',
            'recordSchema': 'MARC21-xml',
            'maximumRecords': '1'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Python/3.13) ThULB/1.0',
            'Accept': 'application/xml'
        }

        delay = self.settings.RETRY_DELAY
        
        async def try_query(query_params):
            if not self.session:
                raise RuntimeError("Session not initialized")
                
            async with self.session.get(
                str(self.settings.DNB_SRU_URL), 
                params=query_params,
                headers=headers
            ) as response:
                response.raise_for_status()
                text = await response.text()
                if '<numberOfRecords>0</numberOfRecords>' not in text:
                    return text
                return None

        # Versuche verschiedene Suchstrategien
        for attempt in range(self.settings.RETRY_ATTEMPTS):
            try:
                # Versuch 1: Volle ISBN
                result = await try_query(params)
                if result:
                    return result
                    
                # Versuch 2: ISBN ohne Bindestriche
                isbn_clean = isbn.replace('-', '')
                params['query'] = f'num={isbn_clean}'
                result = await try_query(params)
                if result:
                    return result
                    
                # Versuch 3: Die ersten 10 Stellen der ISBN
                if len(isbn_clean) == 13:
                    isbn_prefix = isbn_clean[:-3]
                    params['query'] = f'num={isbn_prefix}'
                    result = await try_query(params)
                    if result:
                        return result
                
                if attempt < self.settings.RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    
            except aiohttp.ClientError as e:
                self.logger.warning(f"DNB query attempt {attempt + 1} failed: {e}")
                if attempt < self.settings.RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise

        self.logger.error(f"All attempts to query DNB for ISBN {isbn} failed")
        return None

    @staticmethod
    def _validate_isbn(isbn: str) -> bool:
        """Erweiterte ISBN-Validierung."""
        if not isbn:
            return False
            
        # Entferne Bindestriche und Leerzeichen
        isbn = isbn.replace('-', '').replace(' ', '')
        
        # Akzeptiere ISBN-10 und ISBN-13
        if len(isbn) not in [10, 13]:
            return False
            
        # Erlaube Zahlen und 'X' am Ende für ISBN-10
        if not all(c.isdigit() or (c.upper() == 'X' and i == len(isbn)-1) 
                  for i, c in enumerate(isbn)):
            return False
            
        return True

    async def _process_metadata(self, xml_data: str) -> Optional[BookMetadata]:
        """Verarbeitet die XML-Metadaten mit verbesserter Suche."""
        try:
            metadata = self._parse_response(xml_data)
            if metadata:
                # Erweiterte Suchterme für Medizin/Pharmakologie
                additional_terms = []
                if any('Schwangerschaft' in subj for subj in metadata.subjects):
                    additional_terms.extend(['Gynäkologie', 'Geburtshilfe', 'YQ'])
                if any('Pharma' in subj for subj in metadata.subjects):
                    additional_terms.extend(['WB', 'XD 4000'])
                metadata.subjects.extend(additional_terms)
                
                await self._enrich_with_rvk_data(metadata)
                return metadata
            return None
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Metadata processing error: {e}", exc_info=True)
            return None

    def _parse_response(self, xml_data: str) -> Optional[BookMetadata]:
        try:
            root = ET.fromstring(xml_data)
            records = root.findall(".//marc:record", self.namespaces)
            if not records:
                return None

            record = records[0]
            metadata = BookMetadata(
                idn=self._extract_field(record, "001"),
                titel=self._extract_title(record),
                authors=self._extract_authors(record),
                publication_year=self._extract_year(record),
                subjects=self._extract_subjects(record),
                classifications=self._extract_classifications(record)
            )
            
            return metadata if self._validate_metadata(metadata) else None
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}", exc_info=True)
            return None

    def _validate_metadata(self, metadata: BookMetadata) -> bool:
        if not metadata.idn or not metadata.titel:
            self.logger.warning("Missing required metadata fields")
            return False
        return True

    def _extract_field(self, record: ET.Element, tag: str, code: str = None) -> str:
        try:
            if code:
                field = record.find(
                    f".//marc:datafield[@tag='{tag}']/marc:subfield[@code='{code}']",
                    self.namespaces
                )
            else:
                field = record.find(
                    f".//marc:controlfield[@tag='{tag}']",
                    self.namespaces
                )
            return field.text.strip() if field is not None and field.text else ""
        except Exception as e:
            self.logger.error(f"Error extracting field {tag}/{code}: {e}")
            return ""

    def _extract_title(self, record: ET.Element) -> str:
        title = self._extract_field(record, "245", 'a')
        subtitle = self._extract_field(record, "245", 'b')
        return f"{title} {subtitle}".strip() if subtitle else title

    def _extract_year(self, record: ET.Element) -> Optional[str]:
        year = self._extract_field(record, "264", 'c')
        if year:
            year_match = re.search(r'\d{4}', year)
            if year_match:
                return year_match.group()
        return None

    def _extract_authors(self, record: ET.Element) -> List[str]:
        authors = []
        try:
            for tag, code in self.marc_fields['authors']:
                fields = record.findall(
                    f".//marc:datafield[@tag='{tag}']/marc:subfield[@code='{code}']",
                    self.namespaces
                )
                authors.extend(field.text.strip() for field in fields if field.text)
        except Exception as e:
            self.logger.error(f"Error extracting authors: {e}")
        return authors

    def _extract_subjects(self, record: ET.Element) -> List[str]:
        subjects = []
        try:
            for tag, code in self.marc_fields['subjects']:
                fields = record.findall(
                    f".//marc:datafield[@tag='{tag}']/marc:subfield[@code='{code}']",
                    self.namespaces
                )
                subjects.extend(field.text.strip() for field in fields if field.text)
        except Exception as e:
            self.logger.error(f"Error extracting subjects: {e}")
        return subjects

    def _extract_classifications(self, record: ET.Element) -> List[str]:
        classifications = []
        try:
            fields = record.findall(
                ".//marc:datafield[@tag='082']/marc:subfield[@code='a']",
                self.namespaces
            )
            classifications.extend(field.text.strip() for field in fields if field.text)
        except Exception as e:
            self.logger.error(f"Error extracting classifications: {e}")
        return classifications

    async def _enrich_with_rvk_data(self, metadata: BookMetadata):
        """Reichert Metadaten mit RVK-Daten an, für alle Fachbereiche."""
        try:
            self.logger.debug(f"Starting RVK enrichment for title: {metadata.titel}")
            
            # Erweiterte Suchtextbasis
            search_text = ' '.join([
                metadata.titel.lower(),
                *[s.lower() for s in metadata.subjects],
                *[str(c).lower() for c in metadata.classifications]
            ])
            
            # DDC-Hauptklasse zur Fachgebietserkennung
            ddc_class = next((c.split('.')[0] for c in metadata.classifications if c), '')
            
            # Fachgebietserkennung
            subject_area = self._determine_subject_area(search_text, ddc_class)
            self.logger.debug(f"Determined subject area: {subject_area}")
            
            # RVK-Notationen basierend auf Fachgebiet
            direct_notations = self._get_rvk_notations(subject_area)
            
            # ThULB-Mapping nur für Medizin und Biochemie
            thulb_descriptions = self._get_thulb_mapping(subject_area)
            
            seen_rvk = set()
            seen_thulb = set()
            
            # RVK-Notationen hinzufügen
            for notation, description in direct_notations.items():
                if notation in seen_rvk:
                    continue
                    
                metadata.rvk_notations.append({
                    'notation': notation,
                    'benennung': description,
                    'source_term': f'Fachgebiet: {subject_area}'
                })
                seen_rvk.add(notation)
            
            # ThULB-Mapping
            if thulb_descriptions:
                for notation, description in thulb_descriptions.items():
                    if notation not in seen_thulb:
                        metadata.thulb_notations.append({
                            'notation': notation,
                            'description': description
                        })
                        seen_thulb.add(notation)
            else:
                metadata.thulb_notations.append({
                    'notation': 'INFO',
                    'description': f'Titel aus dem Fachbereich: {subject_area}. Kein ThULB-Mapping verfügbar.'
                })

            if metadata.rvk_notations:
                self.logger.info(
                    f"Added {len(metadata.rvk_notations)} RVK notations for {subject_area}"
                )
            else:
                self.logger.warning(f"No RVK notations found for title '{metadata.titel}'")

        except Exception as e:
            self.logger.error(f"Error in RVK enrichment: {e}", exc_info=True)
            raise

    def _determine_subject_area(self, search_text: str, ddc_class: str) -> str:
        """Bestimmt den medizinischen Fachbereich mit umfassender Erkennung."""
        search_text = search_text.lower()
        
        # Umfassende Fachbereichsmuster
        subject_patterns = {
            # Grundlagenfächer
            'Biochemie': [
                'biochem', 'molekular', 'stoffwechsel', 'metabol', 'enzym',
                'protein', 'biokatalyse', 'zellbiolog', 'biomolekül',
                'metabolismus', 'biochemisch'
            ],
            'Anatomie': [
                'anatom', 'morpholog', 'histolog', 'zytolog', 'gewebe',
                'körperbau', 'präparat', 'sektion', 'topograph'
            ],
            'Physiologie': [
                'physiolog', 'körperfunktion', 'homöostase', 'regulation',
                'organfunktion', 'zellphysiolog'
            ],
            
            # Klinische Fächer
            'Innere Medizin': [
                'innere medizin', 'internist', 'kardiolog', 'gastroenterolog',
                'pneumolog', 'hämatolog', 'endokrinolog', 'diabetolog',
                'rheumatolog', 'nephrolog'
            ],
            'Chirurgie': [
                'chirurg', 'operat', 'minimal-invasiv', 'transplant',
                'viszer', 'unfallchirurg', 'traumatolog', 'orthopäd'
            ],
            'Neurologie': [
                'neurolog', 'nervenheilkund', 'neurophysiolog', 'nervensystem',
                'neurodegener', 'neuropsych', 'epilep', 'schlaganfall'
            ],
            'Psychiatrie': [
                'psychiatr', 'psychotherap', 'psychosomat', 'psychopharmak',
                'mental', 'psychose', 'depression', 'schizophren'
            ],
            'Dermatologie': [
                'dermatolog', 'haut', 'cutis', 'derm', 'venerolog',
                'allergolog', 'dermatos', 'dermatitis', 'ekzem'
            ],
            'Gynäkologie': [
                'gynäkolog', 'geburtshilf', 'schwanger', 'pränat',
                'obstetri', 'reproduktion', 'fertili'
            ],
            'Pädiatrie': [
                'pädiatr', 'kinderheilkund', 'neonatolog', 'säugling',
                'entwicklungsstör', 'kinderarzt'
            ],
            
            # Diagnostische Fächer
            'Radiologie': [
                'radiolog', 'bildgebung', 'röntgen', 'computertomograph',
                'magnetresonanztomograph', 'nuklearmedizin', 'ultraschall'
            ],
            'Pathologie': [
                'patholog', 'krankheitslehr', 'histopatholog', 'zytopatholog',
                'nekros', 'biops'
            ],
            'Labormedizin': [
                'labormedizin', 'klinische chemie', 'labordiagnost',
                'laboruntersuch', 'laborparameter'
            ],
            
            # Weitere Spezialfächer
            'Mikrobiologie': [
                'mikrobiolog', 'bakteriolog', 'virolog', 'mykolog',
                'parasitolog', 'infektiolog'
            ],
            'Pharmakologie': [
                'pharmakolog', 'arzneimittel', 'pharmazeut', 'pharmakokinetik',
                'pharmakodynamik', 'toxikolog'
            ],
            'Immunologie': [
                'immunolog', 'immunsystem', 'allergie', 'autoimmun',
                'immundefekt', 'impfung'
            ],
            'Notfallmedizin': [
                'notfall', 'intensivmedizin', 'reanimat', 'erste hilfe',
                'rettung', 'emergency'
            ]
        }
        
        # DDC-Klassifikationen für medizinische Fachgebiete
        ddc_mappings = {
            '572': 'Biochemie',
            '611': 'Anatomie',
            '612': 'Physiologie',
            '615': 'Pharmakologie',
            '616.1': 'Innere Medizin',
            '616.5': 'Dermatologie',
            '616.8': 'Neurologie',
            '616.89': 'Psychiatrie',
            '617': 'Chirurgie',
            '618': 'Gynäkologie',
            '618.92': 'Pädiatrie',
            '616.07': 'Pathologie',
            '616.01': 'Mikrobiologie',
            '616.079': 'Immunologie'
        }
        
        # Prüfe zuerst auf spezifische Fachbegriffe
        for subject, patterns in subject_patterns.items():
            if any(pattern in search_text for pattern in patterns):
                return subject
                
        # Wenn keine spezifischen Begriffe gefunden, prüfe DDC
        if ddc_class:
            for ddc_prefix, subject in ddc_mappings.items():
                if ddc_class.startswith(ddc_prefix):
                    return subject
        
        # Fallback für allgemeine medizinische Literatur
        if '610' in str(ddc_class) or 'medizin' in search_text:
            return 'Medizin'
            
        return 'Allgemeines'

    def _get_rvk_notations(self, subject_area: str) -> Dict[str, str]:
        """Liefert spezifische RVK-Notationen für medizinische Fachgebiete."""
        rvk_notations = {
            # Grundlagenfächer
            'Biochemie': {
                'WW 1500': 'Biochemie - Allgemeines',
                'WW 1600': 'Biochemie - Lehrbücher und Kompendien',
                'WW 2000': 'Biochemie - Spezielle Gebiete',
                'WW 3000': 'Biochemische Prozesse und Stoffwechsel',
                'WW 3500': 'Molekularbiologie und Zellbiochemie'
            },
            'Anatomie': {
                'WW 1000': 'Anatomie - Allgemeines',
                'WW 1100': 'Anatomie - Lehrbücher',
                'WW 1200': 'Anatomie - Atlas und Bildwerke',
                'WW 1300': 'Topographische Anatomie',
                'WW 1400': 'Mikroskopische Anatomie'
            },
            'Physiologie': {
                'WW 1460': 'Physiologie - Allgemeines',
                'WW 1500': 'Physiologie - Lehrbücher',
                'WW 1520': 'Physiologie - Spezielle Gebiete',
                'WW 1530': 'Neurophysiologie',
                'WW 1540': 'Stoffwechselphysiologie'
            },
            
            # Klinische Fächer
            'Innere Medizin': {
                'YB 1000': 'Innere Medizin - Allgemeines',
                'YB 2000': 'Innere Medizin - Lehrbücher',
                'YB 3300': 'Innere Medizin - Diagnostik',
                'YB 4000': 'Innere Medizin - Therapie',
                'YB 5000': 'Spezielle Pathologie und Therapie'
            },
            'Chirurgie': {
                'YL 1000': 'Chirurgie - Allgemeines',
                'YL 2000': 'Chirurgische Diagnostik',
                'YL 3000': 'Chirurgische Therapie',
                'YL 4000': 'Spezielle Chirurgie',
                'YL 5000': 'Transplantationschirurgie'
            },
            'Neurologie': {
                'YK 1000': 'Neurologie - Allgemeines',
                'YK 2000': 'Neurologische Diagnostik',
                'YK 3000': 'Neurologische Therapie',
                'YK 4000': 'Spezielle Neurologie',
                'YK 5000': 'Neuroradiologie'
            },
            'Psychiatrie': {
                'YK 7000': 'Psychiatrie - Allgemeines',
                'YK 8000': 'Psychiatrische Diagnostik',
                'YK 9000': 'Psychiatrische Therapie',
                'YK 9500': 'Psychopharmakologie'
            },
            'Dermatologie': {
                'YH 1000': 'Dermatologie - Allgemeines',
                'YH 2000': 'Dermatologische Diagnostik',
                'YH 3000': 'Dermatologische Therapie',
                'YH 4000': 'Spezielle Dermatologie',
                'YH 5000': 'Venerologie'
            },
            'Gynäkologie': {
                'YQ 1000': 'Gynäkologie - Allgemeines',
                'YQ 2000': 'Gynäkologische Diagnostik',
                'YQ 3000': 'Gynäkologische Therapie',
                'YQ 4000': 'Schwangerschaft und Geburtshilfe',
                'YQ 5000': 'Reproduktionsmedizin'
            },
            'Pädiatrie': {
                'YP 1000': 'Pädiatrie - Allgemeines',
                'YP 2000': 'Pädiatrische Diagnostik',
                'YP 3000': 'Pädiatrische Therapie',
                'YP 4000': 'Neonatologie',
                'YP 5000': 'Entwicklungsstörungen'
            },
            
            # Diagnostische Fächer
            'Radiologie': {
                'YN 1000': 'Radiologie - Allgemeines',
                'YN 2000': 'Radiologische Diagnostik',
                'YN 3000': 'Interventionelle Radiologie',
                'YN 4000': 'Nuklearmedizin',
                'YN 5000': 'Strahlentherapie'
            },
            'Pathologie': {
                'WW 4000': 'Pathologie - Allgemeines',
                'WW 4100': 'Pathologie - Lehrbücher',
                'WW 4200': 'Spezielle Pathologie',
                'WW 4300': 'Histopathologie',
                'WW 4400': 'Molekularpathologie'
            },
            'Labormedizin': {
                'YF 1000': 'Labormedizin - Allgemeines',
                'YF 2000': 'Labordiagnostik',
                'YF 3000': 'Klinische Chemie',
                'YF 4000': 'Spezielle Labormedizin'
            },
            
            # Weitere Spezialfächer
            'Mikrobiologie': {
                'WW 5000': 'Mikrobiologie - Allgemeines',
                'WW 5100': 'Bakteriologie',
                'WW 5200': 'Virologie',
                'WW 5300': 'Mykologie',
                'WW 5400': 'Parasitologie'
            },
            'Pharmakologie': {
                'WW 8000': 'Pharmakologie - Allgemeines',
                'WW 8100': 'Pharmakologie - Lehrbücher',
                'WW 8200': 'Spezielle Pharmakologie',
                'WW 8300': 'Pharmakokinetik',
                'WW 8400': 'Toxikologie'
            },
            
            # Fallback für allgemeine Medizin
            'Medizin': {
                'XB 2000': 'Medizin - Lehrbücher und Kompendien',
                'XB 3000': 'Medizin - Spezielle Gebiete',
                'XB 4000': 'Medizin - Forschung und Methodik'
            }
        }
        
        return rvk_notations.get(subject_area, {
            'XB 0000': f'Allgemeines zum Fachgebiet {subject_area}',
            'XB 1000': f'Grundlagen - {subject_area}'
        })

    def _get_thulb_mapping(self, subject_area: str) -> Optional[Dict[str, str]]:
        """Liefert ThULB-Mappings für medizinische Fachgebiete."""
        medical_mappings = {
            # Grundlagenfächer
            'Biochemie': {
                'MED GC': 'Biochemie und Molekularbiologie'
            },
            'Anatomie': {
                'MED GA': 'Anatomie und Morphologie'
            },
            'Physiologie': {
                'MED GC': 'Physiologie und Biochemie'
            },
            
            # Klinische Fächer
            'Innere Medizin': {
                'MED UI': 'Innere Medizin'
            },
            'Chirurgie': {
                'MED UL': 'Chirurgie und operative Medizin'
            },
            'Neurologie': {
                'MED UK': 'Neurologie und Neurochirurgie'
            },
            'Psychiatrie': {
                'MED UK': 'Psychiatrie und Psychotherapie'
            },
            'Dermatologie': {
                'MED UH': 'Dermatologie und Venerologie'
            },
            'Gynäkologie': {
                'MED UG': 'Gynäkologie und Geburtshilfe'
            },
            'Pädiatrie': {
                'MED UN': 'Kinderheilkunde'
            },
            
            # Diagnostische Fächer
            'Radiologie': {
                'MED UO': 'Radiologie und bildgebende Verfahren'
            },
            'Pathologie': {
                'MED GH': 'Pathologie und pathologische Anatomie'
            },
            'Labormedizin': {
                'MED GC': 'Labormedizin und klinische Chemie'
            },
            
            # Weitere Spezialfächer
            'Mikrobiologie': {
                'MED UE': 'Mikrobiologie und Infektionskrankheiten'
            },
            'Pharmakologie': {
                'MED P': 'Pharmakologie und Toxikologie'
            },
            'Immunologie': {
                'MED UE': 'Immunologie und Allergologie'
            },
            'Notfallmedizin': {
                'MED UL': 'Notfall- und Intensivmedizin'
            },
            
            # Allgemeine Medizin
            'Medizin': {
                'MED A': 'Medizin allgemein'
            }
        }
        
        mapping = medical_mappings.get(subject_area)
        if mapping:
            return mapping
        else:
            # Hinweis für nicht-medizinische Fachgebiete
            return {
                'INFO': f'Fachgebiet {subject_area}: Kein spezifisches ThULB-Mapping verfügbar'
            }

    def _validate_mapping(self, metadata: BookMetadata, subject_area: str):
        """Validiert und ergänzt die Mappings."""
        self.logger.debug(f"Validating mappings for subject area: {subject_area}")
        
        # Prüfe auf fehlende Notationen
        if not metadata.rvk_notations:
            self.logger.warning(f"No RVK notations found for {subject_area}")
            metadata.rvk_notations.append({
                'notation': 'XB 0000',
                'benennung': f'Allgemeines - {subject_area}',
                'source_term': 'Fallback'
            })
            
        if not metadata.thulb_notations:
            self.logger.warning(f"No ThULB notations found for {subject_area}")
            if subject_area in ['Medizin', 'Biochemie']:
                metadata.thulb_notations.append({
                    'notation': 'MED A',
                    'description': 'Medizin allgemein (Fallback)'
                })
                
        # Logge die gefundenen Notationen
        self.logger.info(
            f"Final mappings for {subject_area}: "
            f"{len(metadata.rvk_notations)} RVK, "
            f"{len(metadata.thulb_notations)} ThULB"
        )

def setup_logging(log_file: str = "library_classifier.log"):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )

async def process_isbn_file(
    file_path: str,
    processor: DNBProcessor,
    output_file: str = "results.json"
) -> None:
    logger = logging.getLogger(__name__)
    
    try:
        with open(file_path, 'r') as f:
            isbns = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(isbns)} ISBNs from {file_path}")
        
        results = await processor.process_isbn_batch(isbns)
        
        serializable_results = {}
        for isbn, metadata in results.items():
            if metadata:
                serializable_results[isbn] = {
                    'idn': metadata.idn,
                    'titel': metadata.titel,
                    'authors': metadata.authors,
                    'publication_year': metadata.publication_year,
                    'subjects': metadata.subjects,
                    'classifications': metadata.classifications,
                    'rvk_notations': metadata.rvk_notations,
                    'thulb_notations': metadata.thulb_notations
                }
            else:
                serializable_results[isbn] = None
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        success_count = sum(1 for result in results.values() if result is not None)
        logger.info(f"Processing complete: {success_count}/{len(isbns)} successful")
        
    except Exception as e:
        logger.error(f"Error processing ISBN file: {e}", exc_info=True)
        raise

async def main():
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting library classification process")
        
        settings = Settings()
        logger.info(f"Loaded settings: {settings}")
        
        rvk_client = RVKClient(settings)
        mapper = RVKThULBMapper()
        
        async with DNBProcessor(settings, rvk_client, mapper) as processor:
            isbn = "978-3-437-21203-1"
            logger.info(f"Processing single ISBN: {isbn}")
            
            result = await processor.process_isbn(isbn)
            if result:
                print("\nErgebnisse für ISBN", isbn)
                print(f"Titel: {result.titel}")
                if result.authors:
                    print(f"Autoren: {', '.join(result.authors)}")
                if result.publication_year:
                    print(f"Erscheinungsjahr: {result.publication_year}")
                
                if result.subjects:
                    print("\nSchlagwörter:")
                    for subject in result.subjects:
                        print(f"- {subject}")
                
                if result.rvk_notations:
                    print("\nRVK-Klassifikationen:")
                    for rvk in result.rvk_notations:
                        print(f"- {rvk['notation']}: {rvk['benennung']}")
                        if 'source_term' in rvk:
                            print(f"  (gefunden durch Term: {rvk['source_term']})")
                
                if result.thulb_notations:
                    print("\nThULB-Klassifikationen:")
                    seen_notations = set()
                    for thulb in result.thulb_notations:
                        if thulb['notation'] not in seen_notations:
                            print(f"- {thulb['notation']}: {thulb['description']}")
                            seen_notations.add(thulb['notation'])
            else:
                print(f"Keine Ergebnisse für ISBN {isbn} gefunden.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise
    finally:
        logger.info("Library classification process completed")

if __name__ == "__main__":
    asyncio.run(main())