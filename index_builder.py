# legal_index_builder.py

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

# ─── Main Index Builder Function ────────────────────────────────────────────────
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

    parser = argparse.ArgumentParser(description="Legal Document Index Builder")
    parser.add_argument("--pdf_dir", default="pdfs", help="PDF directory")
    parser.add_argument("--index_dir", default="pageindices1", help="PageIndex output directory")
    args = parser.parse_args()

    print("Building Enhanced PageIndex system...")
    build_enhanced_pageindex_system(args.pdf_dir, args.index_dir)
    print("\nIndex building completed!")
