# enhanced_legal_rag_with_query_classification.py

import os
import glob
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF parsing
from groq import Groq

# ─── Load environment and initialize clients ────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ─── Enhanced Query Classification System ────────────────────────────────────────
class LegalQueryClassifier:
    def __init__(self):
        self.client = client
    
    def classify_query_with_llm(self, query: str) -> Dict:
        """Enhanced LLM-based query classification"""
        classification_prompt = f"""
Analyze this legal query and classify it. Return a JSON object with the following structure:

{{
  "type": "judge_query|case_query|general_legal|entity_search|count_query",
  "intent": "find_judge|find_cases|list_cases|count_cases|list_judges_in_case|information",
  "entities": {{
    "judge": "extracted judge name if any",
    "case": "extracted case name if any",
    "entity": "extracted entity name if any"
  }},
  "response_format": "simple|detailed|sources_only",
  "confidence": 0.0-1.0
}}

Query: "{query}"

Classification rules:
- judge_query: Questions asking for a specific judge in a case (e.g., "who was the judge in X case")
- count_query: Questions asking for all judgments/cases of a particular judge (e.g., "list all cases of judge X", "how many cases did judge X handle")
- case_query: Questions about specific cases
- general_legal: General legal questions
- entity_search: Search for specific legal entities
- list_judges_in_case: Questions asking for all judges in a particular case

Response format rules:
- simple: For judge_query - return only judge name and source
- detailed: For general queries - return comprehensive answer
- sources_only: For count_query - return only sources/case names

Intent rules:
- find_judge: Single judge identification
- count_cases: Count or list all cases of a judge
- list_judges_in_case: List all judges in a specific case
- information: General information requests

Return only the JSON object.
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    return self._get_default_classification()
            else:
                return self._get_default_classification()
                
        except Exception as e:
            print(f"LLM classification failed: {str(e)}")
            return self._get_default_classification()
    
    def _get_default_classification(self) -> Dict:
        """Return default classification"""
        return {
            "type": "general_legal",
            "intent": "information",
            "entities": {},
            "response_format": "detailed",
            "confidence": 0.5
        }

# ─── Enhanced Legal Entity Extraction with PDF Source ──────────────────────────
class EnhancedLegalEntityExtractor:
    def __init__(self):
        self.client = client
        
    def extract_entities_with_patterns(self, text: str, pdf_source: str) -> Dict:
        """Extract entities using both pattern matching and LLM"""
        pattern_entities = self._extract_with_patterns(text, pdf_source)
        llm_entities = self._extract_with_llm(text, pdf_source)
        merged_entities = self._merge_extractions(pattern_entities, llm_entities)
        return merged_entities
    
    def _extract_with_patterns(self, text: str, pdf_source: str) -> Dict:
        """Extract entities using regex patterns"""
        entities = {
            "source_pdf": pdf_source,
            "plaintiffs": [],
            "defendants": [],
            "appellants": [],
            "respondents": [],
            "plaintiff_advocates": [],
            "defendant_advocates": [],
            "appellant_advocates": [],
            "respondent_advocates": [],
            "judges": [],
            "case_numbers": [],
            "dhc_numbers": [],
            "ia_numbers": [],
            "decision_date": None,
            "order_date": None,
            "damages_claimed": None,
            "damages_awarded": None
        }
        
        # Extract plaintiffs
        plaintiff_pattern = r'PLAINTIFF\s*-\s*([^-]+?)(?=DEFENDANT|PLAINTIFF ADVOCATES|$)'
        plaintiff_match = re.search(plaintiff_pattern, text, re.IGNORECASE | re.DOTALL)
        if plaintiff_match:
            plaintiffs = [p.strip() for p in plaintiff_match.group(1).split('\n') if p.strip()]
            entities["plaintiffs"] = plaintiffs
        
        # Extract defendants
        defendant_pattern = r'DEFENDANT\s*-\s*([^-]+?)(?=PLAINTIFF ADVOCATES|DEFENDANT ADVOCATES|CORAM|$)'
        defendant_match = re.search(defendant_pattern, text, re.IGNORECASE | re.DOTALL)
        if defendant_match:
            defendants = [d.strip() for d in defendant_match.group(1).split('\n') if d.strip()]
            entities["defendants"] = defendants
        
        # Extract appellants
        appellant_pattern = r'APPELLANT\s*-\s*([^-]+?)(?=RESPONDENT|APPELLANT ADVOCATES|$)'
        appellant_match = re.search(appellant_pattern, text, re.IGNORECASE | re.DOTALL)
        if appellant_match:
            appellants = [a.strip() for a in appellant_match.group(1).split('\n') if a.strip()]
            entities["appellants"] = appellants
        
        # Extract respondents
        respondent_pattern = r'RESPONDENT\s*-\s*([^-]+?)(?=APPELLANT ADVOCATES|RESPONDENT ADVOCATES|CORAM|$)'
        respondent_match = re.search(respondent_pattern, text, re.IGNORECASE | re.DOTALL)
        if respondent_match:
            respondents = [r.strip() for r in respondent_match.group(1).split('\n') if r.strip()]
            entities["respondents"] = respondents
        
        # Extract plaintiff advocates
        plt_adv_pattern = r'PLAINTIFF ADVOCATES?\s*-\s*([^-]+?)(?=DEFENDANT ADVOCATES|CORAM|$)'
        plt_adv_match = re.search(plt_adv_pattern, text, re.IGNORECASE | re.DOTALL)
        if plt_adv_match:
            advocates_text = plt_adv_match.group(1).strip()
            advocates = self._parse_advocates(advocates_text)
            entities["plaintiff_advocates"] = advocates
        
        # Extract defendant advocates
        def_adv_pattern = r'DEFENDANT ADVOCATES?\s*-\s*([^-]+?)(?=CORAM|$)'
        def_adv_match = re.search(def_adv_pattern, text, re.IGNORECASE | re.DOTALL)
        if def_adv_match:
            advocates_text = def_adv_match.group(1).strip()
            advocates = self._parse_advocates(advocates_text)
            entities["defendant_advocates"] = advocates
        
        # Extract appellant advocates
        app_adv_pattern = r'APPELLANT ADVOCATES?\s*-\s*([^-]+?)(?=RESPONDENT ADVOCATES|CORAM|$)'
        app_adv_match = re.search(app_adv_pattern, text, re.IGNORECASE | re.DOTALL)
        if app_adv_match:
            advocates_text = app_adv_match.group(1).strip()
            advocates = self._parse_advocates(advocates_text)
            entities["appellant_advocates"] = advocates
        
        # Extract respondent advocates
        res_adv_pattern = r'RESPONDENT ADVOCATES?\s*-\s*([^-]+?)(?=CORAM|$)'
        res_adv_match = re.search(res_adv_pattern, text, re.IGNORECASE | re.DOTALL)
        if res_adv_match:
            advocates_text = res_adv_match.group(1).strip()
            advocates = self._parse_advocates(advocates_text)
            entities["respondent_advocates"] = advocates
        
        # Extract judges from CORAM section
        coram_pattern = r'CORAM:\s*([^:]+?)(?=\n\n|\n[A-Z]{2,}|$)'
        coram_match = re.search(coram_pattern, text, re.IGNORECASE | re.DOTALL)
        if coram_match:
            judges_text = coram_match.group(1).strip()
            judges = self._parse_judges(judges_text)
            entities["judges"] = judges
        
        # Extract case numbers
        case_patterns = [
            r'CS\(COMM\)\s*(\d+/\d+)',
            r'C\.A\.\(COMM\.IPD-TM\)\s*(\d+/\d+)',
            r'I\.A\.\s*(\d+/\d+)',
            r'(\d{4}/DHC/\d+)'
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if 'DHC' in pattern:
                entities["dhc_numbers"].extend(matches)
            elif 'I.A.' in pattern:
                entities["ia_numbers"].extend(matches)
            else:
                entities["case_numbers"].extend(matches)
        
        # Extract dates
        date_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})'
        date_matches = re.findall(date_pattern, text)
        for match in date_matches:
            date_str = f"{match[0]} {match[1]}, {match[2]}"
            if not entities["decision_date"]:
                entities["decision_date"] = date_str
        
        # Extract financial amounts
        amount_pattern = r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        amount_matches = re.findall(amount_pattern, text)
        if amount_matches:
            entities["damages_claimed"] = amount_matches[0]
        
        return entities
    
    def _parse_advocates(self, advocates_text: str) -> List[str]:
        """Parse advocates text handling multiple names and formats"""
        advocates = []
        advocates_text = re.sub(r'\s+', ' ', advocates_text).strip()
        parts = re.split(r'\s+(?:with|and|,)\s+', advocates_text)
        
        for part in parts:
            part = part.strip().rstrip(',')
            if part:
                clean_name = re.sub(r'^(Dr\.|Mr\.|Ms\.|Mrs\.|Adv\.|Advocate)\s+', '', part)
                if clean_name:
                    advocates.append(clean_name)
        
        return advocates
    
    def _parse_judges(self, judges_text: str) -> List[str]:
        """Parse judges text handling multiple judges and titles"""
        judges = []
        judges_text = re.sub(r'\s+', ' ', judges_text).strip()
        
        # Remove common titles and prefixes
        judges_text = re.sub(r'HON\'BLE\s+', '', judges_text, flags=re.IGNORECASE)
        judges_text = re.sub(r'JUSTICE\s+', '', judges_text, flags=re.IGNORECASE)
        judges_text = re.sub(r'MR\.\s+', '', judges_text, flags=re.IGNORECASE)
        judges_text = re.sub(r'MS\.\s+', '', judges_text, flags=re.IGNORECASE)
        judges_text = re.sub(r'CHIEF\s+', '', judges_text, flags=re.IGNORECASE)
        
        parts = re.split(r'\s+(?:and|,)\s+', judges_text)
        
        for part in parts:
            part = part.strip().rstrip(',')
            if part and len(part) > 2:
                judges.append(part)
        
        return judges
    
    def _extract_with_llm(self, text: str, pdf_source: str) -> Dict:
        """Extract entities using LLM as backup/enhancement"""
        prompt = f"""
        Extract legal entities from this document from PDF source: {pdf_source}
        
        Focus on:
        1. Plaintiffs/Appellants
        2. Defendants/Respondents  
        3. Their advocates/lawyers
        4. Judges (from CORAM section)
        5. Case numbers
        6. Important dates
        7. Financial amounts
        
        Text: {text[:4000]}...
        
        Return JSON with these fields:
        {{
            "source_pdf": "{pdf_source}",
            "plaintiffs": [],
            "defendants": [],
            "appellants": [],
            "respondents": [],
            "plaintiff_advocates": [],
            "defendant_advocates": [],
            "appellant_advocates": [],
            "respondent_advocates": [],
            "judges": [],
            "case_numbers": [],
            "dhc_numbers": [],
            "decision_date": null,
            "damages_claimed": null
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    return self._get_empty_entities(pdf_source)
            else:
                return self._get_empty_entities(pdf_source)
                
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return self._get_empty_entities(pdf_source)
    
    def _merge_extractions(self, pattern_entities: Dict, llm_entities: Dict) -> Dict:
        """Merge pattern and LLM extractions, prioritizing pattern results"""
        merged = pattern_entities.copy()
        
        for key, value in llm_entities.items():
            if key in merged:
                if isinstance(value, list):
                    if not merged[key] and value:
                        merged[key] = value
                else:
                    if not merged[key] and value:
                        merged[key] = value
        
        return merged
    
    def _get_empty_entities(self, pdf_source: str) -> Dict:
        """Return empty entity structure with PDF source"""
        return {
            "source_pdf": pdf_source,
            "plaintiffs": [],
            "defendants": [],
            "appellants": [],
            "respondents": [],
            "plaintiff_advocates": [],
            "defendant_advocates": [],
            "appellant_advocates": [],
            "respondent_advocates": [],
            "judges": [],
            "case_numbers": [],
            "dhc_numbers": [],
            "ia_numbers": [],
            "decision_date": None,
            "order_date": None,
            "damages_claimed": None,
            "damages_awarded": None
        }

# ─── Enhanced Legal Document Parser with PDF Source ─────────────────────────────
class ComprehensiveLegalParser:
    def __init__(self):
        self.entity_extractor = EnhancedLegalEntityExtractor()
        
    def extract_full_text(self, pdf_path: str) -> str:
        """Extract complete text from PDF"""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += f"\n--- PAGE {page_num + 1} ---\n"
            full_text += page.get_text()
        doc.close()
        return full_text
    
    def extract_case_details(self, text: str) -> Dict:
        """Extract case details"""
        details = {
            'court_type': 'HIGH COURT OF DELHI AT NEW DELHI',
            'case_type': None,
            'subject_matter': None
        }
        
        if 'trademark' in text.lower():
            details['case_type'] = 'Trademark'
        elif 'copyright' in text.lower():
            details['case_type'] = 'Copyright'
        elif 'patent' in text.lower():
            details['case_type'] = 'Patent'
        elif 'injunction' in text.lower():
            details['case_type'] = 'Injunction'
        elif 'commercial' in text.lower():
            details['case_type'] = 'Commercial'
        else:
            details['case_type'] = 'Civil'
        
        return details
    
    def extract_key_issues_enhanced(self, text: str) -> List[str]:
        """Enhanced key issues extraction"""
        prompt = f"""Extract the main legal issues from this document:
        
        {text[:4000]}...
        
        List 5-10 key legal issues, each on a separate line without numbering."""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            issues_text = response.choices[0].message.content.strip()
            issues_list = [issue.strip() for issue in issues_text.split('\n') if issue.strip()]
            return issues_list[:10]
        except:
            return []
    
    def generate_comprehensive_summary(self, text: str) -> str:
        """Generate comprehensive decision summary"""
        prompt = f"""Provide a comprehensive summary of this legal case:
        
        {text[:5000]}...
        
        Include background, legal issues, arguments, court findings, and decision."""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except:
            return "Summary generation failed"
    
    def extract_legal_precedents(self, text: str) -> List[str]:
        """Extract cited legal precedents"""
        precedents = []
        
        case_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+vs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',
            r'\(\d{4}\)\s+\d+\s+DLT\s+\d+',
            r'AIR\s+\d{4}\s+SC\s+\d+'
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, text)
            precedents.extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        return list(set(precedents))[:10]
    
    def parse_complete_document(self, pdf_path: str, doc_name: str) -> Dict:
        """Main parsing function with enhanced entity extraction and PDF source"""
        full_text = self.extract_full_text(pdf_path)
        
        # Extract PDF source name
        pdf_source = Path(pdf_path).name
        
        print(f"Extracting entities for {doc_name} from PDF: {pdf_source}")
        entities = self.entity_extractor.extract_entities_with_patterns(full_text, pdf_source)
        
        case_details = self.extract_case_details(full_text)
        key_issues = self.extract_key_issues_enhanced(full_text)
        decision_summary = self.generate_comprehensive_summary(full_text)
        precedents = self.extract_legal_precedents(full_text)
        
        metadata = {
            'document_name': doc_name,
            'source_pdf': pdf_source,
            'judges': entities.get('judges', []),
            'court_name': case_details['court_type'],
            'case_numbers': entities.get('case_numbers', []),
            'dhc_numbers': entities.get('dhc_numbers', []),
            'ia_numbers': entities.get('ia_numbers', []),
            'decision_date': entities.get('decision_date'),
            'order_date': entities.get('order_date'),
            'case_type': case_details['case_type'],
            'subject_matter': case_details['subject_matter'],
            'plaintiffs': entities.get('plaintiffs', []),
            'defendants': entities.get('defendants', []),
            'appellants': entities.get('appellants', []),
            'respondents': entities.get('respondents', []),
            'plaintiff_advocates': entities.get('plaintiff_advocates', []),
            'defendant_advocates': entities.get('defendant_advocates', []),
            'appellant_advocates': entities.get('appellant_advocates', []),
            'respondent_advocates': entities.get('respondent_advocates', []),
            'damages_claimed': entities.get('damages_claimed'),
            'damages_awarded': entities.get('damages_awarded'),
            'key_issues': key_issues,
            'decision_summary': decision_summary,
            'legal_precedents': precedents,
            'full_text': full_text
        }
        
        return metadata

# ─── Enhanced Legal Response Generator with PDF Source ──────────────────────────
class LegalResponseGenerator:
    def __init__(self, collection_name: str = "legal_cases"):
        self.collection_name = collection_name
        self.client = client
    
    def generate_simple_response(self, judge_name: str, source_pdf: str) -> str:
        """Generate simple response with just judge name and source"""
        return f"**Judge:** {judge_name}\n**Source:** {source_pdf}"
    
    def generate_sources_only_response(self, judge_name: str, case_sources: List[Dict]) -> str:
        """Generate response with only sources for count queries"""
        if not case_sources:
            return f"No cases found for Judge {judge_name}"
        
        response = f"**Judge {judge_name} - All Cases ({len(case_sources)} total):**\n\n"
        for i, case in enumerate(case_sources, 1):
            response += f"{i}. **Source:** {case['source_pdf']}\n"
            response += f"   **Case:** {case['case_name']}\n"
            if case.get('case_numbers'):
                response += f"   **Case Numbers:** {', '.join(case['case_numbers'])}\n"
            if case.get('decision_date'):
                response += f"   **Date:** {case['decision_date']}\n"
            response += "\n"
        
        return response
    
    def generate_judges_list_response(self, case_name: str, judges: List[str], source_pdf: str) -> str:
        """Generate response listing all judges in a case"""
        if not judges:
            return f"No judges found for case: {case_name}"
        
        response = f"**Judges in {case_name}:**\n\n"
        for i, judge in enumerate(judges, 1):
            response += f"{i}. {judge}\n"
        response += f"\n**Source:** {source_pdf}"
        
        return response
    
    def generate_comprehensive_response(self, query: str, context: str, metadata: Dict = None) -> str:
        """Generate comprehensive legal response with PDF source references"""
        
        # Limit context length to avoid token overflow
        total_context_length = 4000
        context_snippet = context[:total_context_length]
        
        # Enhanced system prompt for legal expertise with PDF source awareness
        system_prompt = f"""You are a highly knowledgeable legal research assistant with expertise in Indian law, case analysis, and legal document interpretation.

Your task is to provide comprehensive, accurate, and well-structured answers based on the provided legal documents from collection '{self.collection_name}'.

Guidelines:
1. Use ONLY the information provided in the context documents
2. Be specific and detailed in your explanations
3. ALWAYS reference specific PDF source files and page numbers when making points
4. If the query asks about a specific case, provide comprehensive details including parties, legal issues, court decisions, and legal principles
5. Organize your response with clear headings and structure
6. Include PDF source file names prominently in your response
7. If information is limited, clearly state what is available and what would require additional research
8. Always maintain professional legal language and accuracy"""
        
        # Enhanced user prompt with PDF source emphasis
        user_prompt = f"""Based on the provided legal documents, please answer this query comprehensively: "{query}"

Context Documents from Collection '{self.collection_name}':

{context_snippet}

Please provide a detailed, well-structured response that:
1. Directly addresses the question asked
2. Uses specific information from the documents provided
3. PROMINENTLY references PDF source files and page numbers throughout the response
4. Provides comprehensive legal analysis where applicable
5. Maintains accuracy and professional legal standards
6. Includes a clear indication of which PDF file each piece of information comes from

Structure your response with clear headings and professional formatting, and make sure to cite the PDF source file for each major point."""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating comprehensive response: {str(e)}"

# ─── Enhanced PageIndex Tree Generator with PDF Source ──────────────────────────
class PageIndexTreeGenerator:
    def __init__(self):
        self.node_counter = 0
        
    def generate_node_id(self) -> str:
        """Generate unique node ID"""
        self.node_counter += 1
        return f"{self.node_counter:04d}"
    
    def extract_page_content(self, pdf_path: str, page_num: int) -> str:
        """Extract text content from a specific page"""
        doc = fitz.open(pdf_path)
        if page_num <= len(doc):
            page = doc[page_num - 1]
            content = page.get_text()
            doc.close()
            return content.strip()
        doc.close()
        return ""
    
    def generate_page_summary(self, page_content: str, page_num: int, pdf_source: str, document_context: str = "") -> str:
        """Generate summary for a specific page with PDF source"""
        if not page_content.strip():
            return f"Page {page_num} from {pdf_source} - No extractable content"
        
        prompt = f"""Summarize page {page_num} from PDF: {pdf_source}
        
        Context: {document_context[:500]}...
        
        Page content: {page_content[:3000]}...
        
        Focus on legal facts, proceedings, arguments, decisions, dates, names, and amounts.
        Keep under 200 words and mention the PDF source."""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Page {page_num} from {pdf_source} - Summary generation failed"
    
    def generate_pageindex_tree(self, pdf_path: str, doc_name: str) -> Dict:
        """Generate complete PageIndex tree structure with PDF source"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Extract PDF source name
        pdf_source = Path(pdf_path).name
        
        full_text = ""
        doc = fitz.open(pdf_path)
        for page_num in range(min(3, total_pages)):
            page = doc[page_num]
            full_text += page.get_text()
        doc.close()
        
        self.node_counter = 0
        sections = self.detect_sections(pdf_path)
        tree_nodes = []
        
        for section in sections:
            section_content = ""
            doc = fitz.open(pdf_path)
            for page_num in range(section['start_page'] - 1, min(section['end_page'], total_pages)):
                if page_num < len(doc):
                    section_content += doc[page_num].get_text()
            doc.close()
            
            section_summary = self.generate_section_summary(
                section_content, section['title'], pdf_source, full_text[:1000]
            )
            
            page_nodes = []
            for page_num in range(section['start_page'], section['end_page'] + 1):
                if page_num <= total_pages:
                    page_content = self.extract_page_content(pdf_path, page_num)
                    page_summary = self.generate_page_summary(
                        page_content, page_num, pdf_source, full_text[:500]
                    )
                    
                    page_node = {
                        "title": f"Page {page_num}",
                        "node_id": self.generate_node_id(),
                        "start_index": page_num,
                        "end_index": page_num,
                        "summary": page_summary,
                        "content": page_content,
                        "pdf_source": pdf_source,
                        "nodes": []
                    }
                    page_nodes.append(page_node)
            
            section_node = {
                "title": section['title'],
                "node_id": self.generate_node_id(),
                "start_index": section['start_page'],
                "end_index": section['end_page'],
                "summary": section_summary,
                "pdf_source": pdf_source,
                "nodes": page_nodes
            }
            tree_nodes.append(section_node)
        
        return {
            "doc_name": doc_name,
            "pdf_source": pdf_source,
            "total_pages": total_pages,
            "tree_structure": tree_nodes
        }
    
    def detect_sections(self, pdf_path: str) -> List[Dict]:
        """Detect document sections and headings"""
        doc = fitz.open(pdf_path)
        sections = []
        
        toc = doc.get_toc()
        if toc:
            for level, title, page_num in toc:
                sections.append({
                    'title': title,
                    'level': level,
                    'start_page': page_num,
                    'end_page': None
                })
        else:
            total_pages = len(doc)
            
            if total_pages <= 5:
                sections = [
                    {'title': 'Case Header and Parties', 'level': 1, 'start_page': 1, 'end_page': 1},
                    {'title': 'Facts and Arguments', 'level': 1, 'start_page': 2, 'end_page': max(2, total_pages-1)},
                    {'title': 'Decision and Orders', 'level': 1, 'start_page': total_pages, 'end_page': total_pages}
                ]
            else:
                sections = [
                    {'title': 'Case Header and Parties', 'level': 1, 'start_page': 1, 'end_page': 2},
                    {'title': 'Background and Facts', 'level': 1, 'start_page': 3, 'end_page': min(5, total_pages//2)},
                    {'title': 'Legal Arguments', 'level': 1, 'start_page': min(6, total_pages//2+1), 'end_page': max(6, total_pages-2)},
                    {'title': 'Court Decision', 'level': 1, 'start_page': max(7, total_pages-1), 'end_page': total_pages}
                ]
        
        for i, section in enumerate(sections):
            if section['end_page'] is None:
                if i + 1 < len(sections):
                    section['end_page'] = sections[i + 1]['start_page'] - 1
                else:
                    section['end_page'] = len(doc)
        
        doc.close()
        return sections
    
    def generate_section_summary(self, content: str, title: str, pdf_source: str, context: str) -> str:
        """Generate summary for a document section with PDF source"""
        if not content.strip():
            return f"Section '{title}' from {pdf_source} - No extractable content"
        
        prompt = f"""Summarize this legal document section from PDF: {pdf_source}
        
        Section: {title}
        Context: {context[:500]}...
        
        Content: {content[:4000]}...
        
        Focus on legal proceedings, facts, arguments, decisions, parties, dates, and amounts.
        Keep detailed but under 300 words and mention the PDF source."""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Section '{title}' from {pdf_source} - Summary generation failed"

# ─── Enhanced Case Matching Functions ───────────────────────────────────────────
def fuzzy_match_case_name(query: str, document_name: str) -> float:
    """Enhanced case name matching with fuzzy logic"""
    query_words = set(query.lower().split())
    doc_words = set(document_name.lower().split())
    
    stop_words = {'case', 'vs', 'versus', 'ltd', 'inc', 'pvt', 'the', 'and', 'of', 'in', 'v', 'v.'}
    query_words -= stop_words
    doc_words -= stop_words
    
    if not query_words:
        return 0.0
    
    intersection = query_words.intersection(doc_words)
    similarity = len(intersection) / len(query_words)
    
    key_terms = ['nri', 'taxi', 'loreal', 'western', 'digital', 'technologies', 'burger', 'king']
    for term in key_terms:
        if term in query.lower() and term in document_name.lower():
            similarity += 0.2
    
    return min(similarity, 1.0)

def find_best_matching_case(query: str, indices: List[Dict]) -> Optional[Dict]:
    """Find the best matching case based on query"""
    best_match = None
    best_score = 0.0
    
    for idx in indices:
        metadata = idx.get('metadata', {})
        doc_name = metadata.get('document_name', '')
        
        score = fuzzy_match_case_name(query, doc_name)
        
        case_numbers = metadata.get('case_numbers', [])
        for case_num in case_numbers:
            if any(word in case_num.lower() for word in query.lower().split()):
                score += 0.3
        
        plaintiffs = metadata.get('plaintiffs', [])
        defendants = metadata.get('defendants', [])
        appellants = metadata.get('appellants', [])
        respondents = metadata.get('respondents', [])
        
        all_parties = plaintiffs + defendants + appellants + respondents
        for party in all_parties:
            if party and fuzzy_match_case_name(query, party) > 0.3:
                score += 0.2
        
        if score > best_score:
            best_score = score
            best_match = idx
    
    return best_match if best_score > 0.3 else None

def find_cases_by_judge(judge_name: str, indices: List[Dict]) -> List[Dict]:
    """Find all cases where a specific judge was involved"""
    matching_cases = []
    
    for idx in indices:
        metadata = idx.get('metadata', {})
        judges = metadata.get('judges', [])
        
        for judge in judges:
            if judge and judge_name.lower() in judge.lower():
                case_info = {
                    'case_name': metadata.get('document_name'),
                    'source_pdf': metadata.get('source_pdf'),
                    'judge_name': judge,
                    'case_numbers': metadata.get('case_numbers', []),
                    'decision_date': metadata.get('decision_date'),
                    'case_type': metadata.get('case_type')
                }
                matching_cases.append(case_info)
                break
    
    return matching_cases

# ─── Enhanced PageIndex Retrieval with PDF Source ──────────────────────────────
class PageIndexRetrieval:
    def __init__(self, indices):
        self.indices = indices
    
    def search_tree_nodes(self, query: str, tree_structure: List[Dict], max_results: int = 5) -> List[Dict]:
        """Search through tree nodes using query relevance"""
        query_lower = query.lower()
        results = []
        
        def traverse_nodes(nodes, parent_path=""):
            for node in nodes:
                relevance_score = 0
                
                summary = node.get('summary', '').lower()
                title = node.get('title', '').lower()
                
                query_terms = query_lower.split()
                for term in query_terms:
                    if term in summary:
                        relevance_score += 2
                    if term in title:
                        relevance_score += 1
                
                if relevance_score > 0:
                    relevant_contents = []
                    if 'content' in node:
                        content = node['content']
                        paragraphs = content.split('\n\n')
                        for para in paragraphs:
                            if any(term in para.lower() for term in query_terms):
                                relevant_contents.append({
                                    "physical_index": node.get('start_index', 0),
                                    "relevant_content": para.strip()[:500] + "..." if len(para) > 500 else para.strip()
                                })
                    
                    result = {
                        "title": node.get('title', ''),
                        "node_id": node.get('node_id', ''),
                        "start_index": node.get('start_index', 0),
                        "end_index": node.get('end_index', 0),
                        "summary": node.get('summary', ''),
                        "relevance_score": relevance_score,
                        "path": parent_path + "/" + node.get('title', ''),
                        "relevant_contents": relevant_contents,
                        "pdf_source": node.get('pdf_source', 'Unknown PDF')
                    }
                    results.append(result)
                
                if 'nodes' in node:
                    traverse_nodes(node['nodes'], parent_path + "/" + node.get('title', ''))
        
        traverse_nodes(tree_structure)
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:max_results]
    
    def retrieve_context(self, query: str, max_results: int = 5) -> Dict:
        """Retrieve relevant context using PageIndex methodology"""
        all_results = []
        
        for index in self.indices:
            doc_name = index.get('doc_name', '')
            tree_structure = index.get('tree_structure', [])
            pdf_source = index.get('pdf_source', 'Unknown PDF')
            
            doc_results = self.search_tree_nodes(query, tree_structure, max_results)
            
            for result in doc_results:
                result['document'] = doc_name
                result['pdf_source'] = pdf_source
                all_results.append(result)
        
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        if all_results:
            response = {
                "title": f"Legal Query: {query}",
                "query": query,
                "total_results": len(all_results),
                "nodes": []
            }
            
            doc_groups = {}
            for result in all_results[:max_results]:
                doc = result['document']
                if doc not in doc_groups:
                    doc_groups[doc] = []
                doc_groups[doc].append(result)
            
            for doc, results in doc_groups.items():
                for result in results:
                    node_data = {
                        "title": result['title'],
                        "node_id": result['node_id'],
                        "document": doc,
                        "start_index": result['start_index'],
                        "end_index": result['end_index'],
                        "summary": result['summary'],
                        "relevance_score": result['relevance_score'],
                        "relevant_contents": result['relevant_contents'],
                        "pdf_source": result['pdf_source']
                    }
                    response["nodes"].append(node_data)
            
            return response
        
        return {
            "title": f"Legal Query: {query}",
            "query": query,
            "total_results": 0,
            "nodes": []
        }

# ─── Enhanced Query Handler with Classification ─────────────────────────────────
class EnhancedLegalQueryHandler:
    def __init__(self, indices):
        self.indices = indices
        self.retrieval_engine = PageIndexRetrieval(indices)
        self.response_generator = LegalResponseGenerator("legal_cases_collection")
        self.query_classifier = LegalQueryClassifier()
    
    def load_indices(self, index_dir):
        """Load all PageIndex files"""
        indices = []
        for idx_file in glob.glob(os.path.join(index_dir, "*_pageindex.json")):
            with open(idx_file, encoding="utf-8") as f:
                indices.append(json.load(f))
        return indices
    
    def get_complete_case_context(self, best_match: Dict) -> str:
        """Get complete case context including all structured data with PDF source"""
        if not best_match:
            return ""
        
        metadata = best_match.get('metadata', {})
        tree_structure = best_match.get('tree_structure', [])
        pdf_source = best_match.get('pdf_source', metadata.get('source_pdf', 'Unknown PDF'))
        
        # Build comprehensive context with PDF source
        context_parts = []
        
        # Add PDF source prominently
        context_parts.append(f"**SOURCE PDF:** {pdf_source}")
        context_parts.append(f"**Document:** {metadata.get('document_name', '')}")
        context_parts.append(f"**Court:** {metadata.get('court_name', '')}")
        context_parts.append(f"**Case Type:** {metadata.get('case_type', '')}")
        
        if metadata.get('judges'):
            context_parts.append(f"**Judges:** {', '.join(metadata.get('judges', []))} (Source: {pdf_source})")
        
        if metadata.get('case_numbers'):
            context_parts.append(f"**Case Numbers:** {', '.join(metadata.get('case_numbers', []))} (Source: {pdf_source})")
        
        if metadata.get('decision_date'):
            context_parts.append(f"**Decision Date:** {metadata.get('decision_date')} (Source: {pdf_source})")
        
        # Add parties with PDF source
        if metadata.get('plaintiffs'):
            context_parts.append(f"**Plaintiffs:** {', '.join(metadata.get('plaintiffs', []))} (Source: {pdf_source})")
        if metadata.get('defendants'):
            context_parts.append(f"**Defendants:** {', '.join(metadata.get('defendants', []))} (Source: {pdf_source})")
        if metadata.get('appellants'):
            context_parts.append(f"**Appellants:** {', '.join(metadata.get('appellants', []))} (Source: {pdf_source})")
        if metadata.get('respondents'):
            context_parts.append(f"**Respondents:** {', '.join(metadata.get('respondents', []))} (Source: {pdf_source})")
        
        # Add advocates with PDF source
        if metadata.get('plaintiff_advocates'):
            context_parts.append(f"**Plaintiff Advocates:** {', '.join(metadata.get('plaintiff_advocates', []))} (Source: {pdf_source})")
        if metadata.get('defendant_advocates'):
            context_parts.append(f"**Defendant Advocates:** {', '.join(metadata.get('defendant_advocates', []))} (Source: {pdf_source})")
        if metadata.get('appellant_advocates'):
            context_parts.append(f"**Appellant Advocates:** {', '.join(metadata.get('appellant_advocates', []))} (Source: {pdf_source})")
        if metadata.get('respondent_advocates'):
            context_parts.append(f"**Respondent Advocates:** {', '.join(metadata.get('respondent_advocates', []))} (Source: {pdf_source})")
        
        # Add key issues with PDF source
        if metadata.get('key_issues'):
            context_parts.append(f"**Key Legal Issues (Source: {pdf_source}):**")
            for issue in metadata.get('key_issues', []):
                context_parts.append(f"- {issue}")
        
        # Add decision summary with PDF source
        if metadata.get('decision_summary'):
            context_parts.append(f"**Decision Summary (Source: {pdf_source}):**")
            context_parts.append(metadata.get('decision_summary'))
        
        # Add legal precedents with PDF source
        if metadata.get('legal_precedents'):
            context_parts.append(f"**Legal Precedents Cited (Source: {pdf_source}):**")
            for precedent in metadata.get('legal_precedents', []):
                context_parts.append(f"- {precedent}")
        
        # Add financial information with PDF source
        if metadata.get('damages_claimed'):
            context_parts.append(f"**Damages Claimed:** Rs. {metadata.get('damages_claimed')} (Source: {pdf_source})")
        if metadata.get('damages_awarded'):
            context_parts.append(f"**Damages Awarded:** Rs. {metadata.get('damages_awarded')} (Source: {pdf_source})")
        
        # Add page-wise content from tree structure with PDF source
        context_parts.append(f"\n**Document Content by Pages (Source: {pdf_source}):**")
        for section in tree_structure:
            context_parts.append(f"\n**{section.get('title', 'Unknown Section')} (Pages {section.get('start_index', 0)}-{section.get('end_index', 0)}) - Source: {pdf_source}:**")
            context_parts.append(section.get('summary', 'No summary available'))
            
            for page_node in section.get('nodes', []):
                context_parts.append(f"\n**{page_node.get('title', 'Unknown Page')} - Source: {pdf_source}:**")
                context_parts.append(page_node.get('summary', 'No summary available'))
        
        return "\n".join(context_parts)
    
    def answer_query(self, query: str) -> str:
        """Answer queries using enhanced classification and routing"""
        # Step 1: Classify the query
        classification = self.query_classifier.classify_query_with_llm(query)
        
        print(f"Query Classification: {classification}")
        
        query_type = classification.get('type', 'general_legal')
        intent = classification.get('intent', 'information')
        response_format = classification.get('response_format', 'detailed')
        entities = classification.get('entities', {})
        
        # Step 2: Route based on classification
        if query_type == 'judge_query' and intent == 'find_judge':
            return self.handle_simple_judge_query(query, entities)
        
        elif query_type == 'count_query' and intent == 'count_cases':
            return self.handle_count_query(query, entities)
        
        elif intent == 'list_judges_in_case':
            return self.handle_list_judges_query(query, entities)
        
        else:
            return self.handle_general_query(query)
    
    def handle_simple_judge_query(self, query: str, entities: Dict) -> str:
        """Handle simple judge queries - return judge name and source only"""
        # Extract case from query
        case_query = query.lower().replace("judge", "").replace("who", "").replace("was", "").replace("the", "").replace("in", "").strip()
        
        best_match = find_best_matching_case(case_query, self.indices)
        
        if best_match:
            metadata = best_match.get('metadata', {})
            judges = metadata.get('judges', [])
            source_pdf = metadata.get('source_pdf', 'Unknown PDF')
            
            if judges:
                judge_name = judges[0]  # Take the first judge
                return self.response_generator.generate_simple_response(judge_name, source_pdf)
            else:
                return f"No judge information found for case: {case_query}"
        else:
            return f"No case found matching: {case_query}"
    
    def handle_count_query(self, query: str, entities: Dict) -> str:
        """Handle count queries - return all cases and sources for a judge"""
        # Extract judge name from query
        judge_name = entities.get('judge', '')
        
        if not judge_name:
            # Try to extract judge name from query text
            judge_name_parts = []
            for word in query.split():
                if word.lower() not in ['list', 'all', 'cases', 'of', 'judge', 'how', 'many', 'did', 'handle', 'involved']:
                    judge_name_parts.append(word)
            judge_name = ' '.join(judge_name_parts).strip()
        
        if not judge_name:
            return "Could not extract judge name from query"
        
        # Find all cases for this judge
        matching_cases = find_cases_by_judge(judge_name, self.indices)
        
        if matching_cases:
            return self.response_generator.generate_sources_only_response(judge_name, matching_cases)
        else:
            return f"No cases found for Judge {judge_name}"
    
    def handle_list_judges_query(self, query: str, entities: Dict) -> str:
        """Handle queries asking for all judges in a case"""
        case_query = entities.get('case', '')
        
        if not case_query:
            # Extract case from query
            case_query = query.lower().replace("judges", "").replace("who", "").replace("were", "").replace("the", "").replace("in", "").strip()
        
        best_match = find_best_matching_case(case_query, self.indices)
        
        if best_match:
            metadata = best_match.get('metadata', {})
            judges = metadata.get('judges', [])
            source_pdf = metadata.get('source_pdf', 'Unknown PDF')
            case_name = metadata.get('document_name', 'Unknown Case')
            
            return self.response_generator.generate_judges_list_response(case_name, judges, source_pdf)
        else:
            return f"No case found matching: {case_query}"
    
    def handle_general_query(self, query: str) -> str:
        """Handle general queries with comprehensive LLM response"""
        retrieval_result = self.retrieval_engine.retrieve_context(query, max_results=5)
        
        if retrieval_result['total_results'] > 0:
            context_parts = []
            for node in retrieval_result['nodes']:
                pdf_source = node.get('pdf_source', 'Unknown PDF')
                context_parts.append(f"Document: {node['document']} (Source: {pdf_source})")
                context_parts.append(f"Section: {node['title']} (Pages {node['start_index']}-{node['end_index']}) - Source: {pdf_source}")
                context_parts.append(f"Summary: {node['summary']}")
                
                if node['relevant_contents']:
                    context_parts.append(f"Relevant Content from {pdf_source}:")
                    for content in node['relevant_contents']:
                        context_parts.append(f"- {content['relevant_content']}")
            
            combined_context = "\n".join(context_parts)
            return self.response_generator.generate_comprehensive_response(query, combined_context)
        else:
            return "No relevant information found for your query."

# ─── Main System Builder with PDF Source ───────────────────────────────────────
def build_enhanced_pageindex_system(pdf_dir, index_dir):
    """Build complete PageIndex system with PDF source tracking"""
    os.makedirs(index_dir, exist_ok=True)
    
    tree_generator = PageIndexTreeGenerator()
    legal_parser = ComprehensiveLegalParser()
    
    for pdf_path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        name = Path(pdf_path).stem
        pdf_source = Path(pdf_path).name
        print(f"\nProcessing: {name} from PDF: {pdf_source}")
        
        try:
            metadata = legal_parser.parse_complete_document(pdf_path, name)
            pageindex_tree = tree_generator.generate_pageindex_tree(pdf_path, name)
            
            output_data = {
                "doc_name": name,
                "pdf_source": pdf_source,
                "metadata": metadata,
                "total_pages": pageindex_tree['total_pages'],
                "tree_structure": pageindex_tree['tree_structure'],
                "created_at": datetime.now().isoformat(),
                "processing_version": "enhanced_pageindex_v5.0_with_classification"
            }
            
            output_path = os.path.join(index_dir, f"{name}_pageindex.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Generated Enhanced PageIndex for {name}")
            print(f"  - Source PDF: {pdf_source}")
            print(f"  - Plaintiffs: {metadata.get('plaintiffs', [])}")
            print(f"  - Defendants: {metadata.get('defendants', [])}")
            print(f"  - Judges: {metadata.get('judges', [])}")
            print(f"  - Case Numbers: {metadata.get('case_numbers', [])}")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

# ─── Main Application ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Legal RAG System with Query Classification")
    parser.add_argument("--pdf_dir", default="pdfs", help="PDF directory")
    parser.add_argument("--index_dir", default="pageindices", help="Enhanced PageIndex directory")
    args = parser.parse_args()

    # Build Enhanced PageIndex system
    print("Building Enhanced PageIndex system with query classification...")
    build_enhanced_pageindex_system(args.pdf_dir, args.index_dir)

    # Load indices and start query handler
    handler = EnhancedLegalQueryHandler([])
    kb = handler.load_indices(args.index_dir)
    handler.indices = kb
    handler.retrieval_engine = PageIndexRetrieval(kb)

    print("\nEnhanced Legal RAG System with Query Classification Ready!")
    print("\nSupported query types:")
    print("- Simple judge queries: 'Who was the judge in Burger King case?' → Returns: Judge name + Source")
    print("- Count queries: 'List all cases of Judge Amit Bansal' → Returns: All cases + Sources only")
    print("- List judges queries: 'Who were the judges in Western Digital case?' → Returns: All judges in case")
    print("- General queries: Any other legal question → Returns: Comprehensive answer")
    print("\nAll responses include PDF source file names!")
    print()
    
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("\n" + handler.answer_query(q) + "\n")
