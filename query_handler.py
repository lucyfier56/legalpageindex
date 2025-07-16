# enhanced_legal_query_handler.py

import os
import glob
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq

# ─── Load environment and initialize clients ────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ─── Enhanced Content Search Engine ─────────────────────────────────────────────
class ContentSearchEngine:
    """Advanced content search across all JSON documents"""
    
    def __init__(self):
        self.client = client
    
    def search_content_across_documents(self, query: str, indices: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """Search through all document content for relevant matches"""
        matching_docs = []
        query_terms = self._extract_query_terms(query)
        
        for doc_idx in indices:
            relevance_score = self._calculate_document_relevance(doc_idx, query_terms)
            
            if relevance_score >= threshold:
                doc_result = {
                    'document': doc_idx,
                    'relevance_score': relevance_score,
                    'metadata': doc_idx.get('metadata', {}),
                    'relevant_content': self._extract_relevant_content(doc_idx, query_terms)
                }
                matching_docs.append(doc_result)
        
        # Sort by relevance score
        matching_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matching_docs
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query"""
        # Remove common words and extract key terms
        stop_words = {'the', 'case', 'explain', 'tell', 'me', 'about', 'what', 'is', 'was', 'were', 'how', 'why', 'when', 'where', 'all', 'list', 'cases', 'advocate', 'advocates'}
        terms = [term.lower().strip() for term in query.split() if term.lower() not in stop_words and len(term) > 2]
        return terms
    
    def _calculate_document_relevance(self, doc_idx: Dict, query_terms: List[str]) -> float:
        """Calculate document relevance based on content analysis"""
        relevance_score = 0.0
        metadata = doc_idx.get('metadata', {})
        
        # Check document name
        doc_name = metadata.get('document_name', '').lower()
        for term in query_terms:
            if term in doc_name:
                relevance_score += 0.4
        
        # Check parties
        all_parties = (metadata.get('plaintiffs', []) + 
                      metadata.get('defendants', []) + 
                      metadata.get('appellants', []) + 
                      metadata.get('respondents', []))
        
        for party in all_parties:
            if party:
                party_lower = party.lower()
                for term in query_terms:
                    if term in party_lower:
                        relevance_score += 0.3
        
        # Check advocates - higher weight for advocate searches
        all_advocates = (metadata.get('plaintiff_advocates', []) + 
                        metadata.get('defendant_advocates', []) + 
                        metadata.get('appellant_advocates', []) + 
                        metadata.get('respondent_advocates', []))
        
        for advocate in all_advocates:
            if advocate:
                advocate_lower = advocate.lower()
                for term in query_terms:
                    if term in advocate_lower:
                        relevance_score += 0.5  # Higher weight for advocate matches
        
        # Check case type
        case_type = metadata.get('case_type', '').lower()
        for term in query_terms:
            if term in case_type:
                relevance_score += 0.2
        
        # Check decision summary
        decision_summary = metadata.get('decision_summary', '').lower()
        for term in query_terms:
            if term in decision_summary:
                relevance_score += 0.2
        
        # Check key issues
        key_issues = metadata.get('key_issues', [])
        for issue in key_issues:
            if issue:
                issue_lower = issue.lower()
                for term in query_terms:
                    if term in issue_lower:
                        relevance_score += 0.15
        
        # Check legal precedents
        precedents = metadata.get('legal_precedents', [])
        for precedent in precedents:
            if precedent:
                precedent_lower = precedent.lower()
                for term in query_terms:
                    if term in precedent_lower:
                        relevance_score += 0.1
        
        # Check tree structure content
        tree_structure = doc_idx.get('tree_structure', [])
        content_relevance = self._search_tree_content(tree_structure, query_terms)
        relevance_score += content_relevance
        
        return min(relevance_score, 1.0)
    
    def _search_tree_content(self, tree_structure: List[Dict], query_terms: List[str]) -> float:
        """Search through tree structure content"""
        content_score = 0.0
        max_content_score = 0.5  # Cap content contribution
        
        def search_nodes(nodes, current_score=0.0):
            for node in nodes:
                # Check node title
                title = node.get('title', '').lower()
                for term in query_terms:
                    if term in title:
                        current_score += 0.05
                
                # Check node summary
                summary = node.get('summary', '').lower()
                for term in query_terms:
                    if term in summary:
                        current_score += 0.03
                
                # Check node content
                content = node.get('content', '').lower()
                for term in query_terms:
                    if term in content:
                        current_score += 0.02
                
                # Recursively search child nodes
                if 'nodes' in node:
                    current_score = search_nodes(node['nodes'], current_score)
            
            return min(current_score, max_content_score)
        
        return search_nodes(tree_structure)
    
    def _extract_relevant_content(self, doc_idx: Dict, query_terms: List[str]) -> List[Dict]:
        """Extract relevant content snippets from document"""
        relevant_content = []
        
        # Extract from metadata
        metadata = doc_idx.get('metadata', {})
        
        # Decision summary
        decision_summary = metadata.get('decision_summary', '')
        if decision_summary and any(term in decision_summary.lower() for term in query_terms):
            relevant_content.append({
                'type': 'decision_summary',
                'content': decision_summary,
                'source_section': 'Case Decision Summary'
            })
        
        # Key issues
        key_issues = metadata.get('key_issues', [])
        for issue in key_issues:
            if issue and any(term in issue.lower() for term in query_terms):
                relevant_content.append({
                    'type': 'key_issue',
                    'content': issue,
                    'source_section': 'Key Legal Issues'
                })
        
        # Legal precedents
        precedents = metadata.get('legal_precedents', [])
        for precedent in precedents:
            if precedent and any(term in precedent.lower() for term in query_terms):
                relevant_content.append({
                    'type': 'legal_precedent',
                    'content': precedent,
                    'source_section': 'Legal Precedents'
                })
        
        # Extract from tree structure
        tree_structure = doc_idx.get('tree_structure', [])
        tree_content = self._extract_tree_content(tree_structure, query_terms)
        relevant_content.extend(tree_content)
        
        return relevant_content
    
    def _extract_tree_content(self, tree_structure: List[Dict], query_terms: List[str]) -> List[Dict]:
        """Extract relevant content from tree structure"""
        tree_content = []
        
        def extract_from_nodes(nodes, path=""):
            for node in nodes:
                node_path = f"{path}/{node.get('title', 'Unknown')}" if path else node.get('title', 'Unknown')
                
                # Check summary
                summary = node.get('summary', '')
                if summary and any(term in summary.lower() for term in query_terms):
                    tree_content.append({
                        'type': 'summary',
                        'content': summary,
                        'source_section': node_path,
                        'page_range': f"Pages {node.get('start_index', 0)}-{node.get('end_index', 0)}"
                    })
                
                # Check content
                content = node.get('content', '')
                if content and any(term in content.lower() for term in query_terms):
                    # Extract relevant paragraph
                    relevant_para = self._extract_relevant_paragraph(content, query_terms)
                    if relevant_para:
                        tree_content.append({
                            'type': 'content',
                            'content': relevant_para,
                            'source_section': node_path,
                            'page_range': f"Pages {node.get('start_index', 0)}-{node.get('end_index', 0)}"
                        })
                
                # Recursively process child nodes
                if 'nodes' in node:
                    extract_from_nodes(node['nodes'], node_path)
        
        extract_from_nodes(tree_structure)
        return tree_content
    
    def _extract_relevant_paragraph(self, content: str, query_terms: List[str]) -> str:
        """Extract most relevant paragraph from content"""
        paragraphs = content.split('\n\n')
        best_paragraph = ""
        max_matches = 0
        
        for paragraph in paragraphs:
            matches = sum(1 for term in query_terms if term in paragraph.lower())
            if matches > max_matches:
                max_matches = matches
                best_paragraph = paragraph
        
        if best_paragraph:
            return best_paragraph[:500] + "..." if len(best_paragraph) > 500 else best_paragraph
        return ""

# ─── Enhanced Query Classification for Content Analysis ─────────────────────────
class AgenticLegalQueryClassifier:
    def __init__(self):
        self.client = client
    
    def classify_query_with_llm(self, query: str) -> Dict:
        """Advanced classification that detects explanation queries and advocate searches"""
        classification_prompt = f"""
You are an expert legal AI assistant that classifies legal queries intelligently. Analyze this query and determine:
1. What type of query it is
2. What information needs to be extracted from legal JSON documents
3. How to filter and present the results

Query: "{query}"

Return a JSON object with this structure:
{{
  "query_type": "explanation_query|filtered_search|specific_entity|comprehensive_list|complex_filter|content_analysis|advocate_search",
  "primary_intent": "explain_case|find_cases|list_advocates|find_judge|get_parties|get_case_details|analyze_content|find_advocate_cases",
  "requires_deep_search": true|false,
  "search_terms": ["extracted key terms for content search"],
  "filters": {{
    "judge": "extracted judge name or null",
    "case_type": "extracted case type or null",
    "case_name": "extracted case name or null",
    "party_type": "plaintiff|defendant|appellant|respondent|all",
    "advocate_type": "plaintiff_advocate|defendant_advocate|appellant_advocate|respondent_advocate|all",
    "advocate_name": "extracted advocate name or null",
    "date_range": "extracted date range or null",
    "court": "extracted court name or null"
  }},
  "response_format": {{
    "type": "explanation|list|detailed|summary|comprehensive|multi_source",
    "include_metadata": true|false,
    "include_content": true|false,
    "group_by": "case_type|judge|date|party_type|advocate_type|source_pdf|relevance|none"
  }},
  "extracted_entities": {{
    "main_subject": "what is the main subject of query",
    "qualifiers": ["list of qualifying conditions"],
    "output_fields": ["list of fields to include in response"]
  }},
  "confidence": 0.0-1.0
}}

Examples:
- "explain the toyota case" → explanation_query with explain_case intent, requires_deep_search=true
- "what happened in burger king case" → explanation_query with explain_case intent
- "tell me about western digital case" → explanation_query with explain_case intent
- "list all trademark cases where judge is amit bansal" → filtered_search with judge filter
- "who is the plaintiff in western digital case" → specific_entity with party_type filter
- "list all cases where Harish Vaidyanathan shankar is the advocate" → advocate_search with find_advocate_cases intent
- "find cases with advocate John Smith" → advocate_search with find_advocate_cases intent

For explanation queries, set requires_deep_search=true and include_content=true.
For advocate searches, set query_type=advocate_search and extract advocate name.

Return only the JSON object.
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=600,
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
    
    def classify_advocate_query(self, query: str) -> Dict:
        """Classify queries specifically for advocate searches"""
        query_lower = query.lower()
        
        # Extract advocate name from query
        advocate_name = None
        
        # Enhanced name extraction patterns
        name_patterns = [
            r"harish\s+vaidyanathan\s+shankar",
            r"harish\s+vaidyanathan",
            r"vaidyanathan\s+shankar",
            r"harish.*shankar",
            r"advocate\s+([a-zA-Z\s]+)",
            r"([a-zA-Z\s]+)\s+is\s+the\s+advocate",
            r"([a-zA-Z\s]+)\s+advocate"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if "harish" in match.group(0) or "vaidyanathan" in match.group(0) or "shankar" in match.group(0):
                    advocate_name = "Harish Vaidyanathan"
                    break
                elif hasattr(match, 'group') and len(match.groups()) > 0:
                    advocate_name = match.group(1).strip()
                    break
        
        # If no specific pattern matched, try to extract from context
        if not advocate_name:
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ["harish", "vaidyanathan", "shankar"]:
                    # Build name from surrounding words
                    name_parts = []
                    start = max(0, i-2)
                    end = min(len(words), i+3)
                    for j in range(start, end):
                        if words[j].lower() not in ["is", "the", "advocate", "all", "cases", "where", "list"]:
                            name_parts.append(words[j])
                    if name_parts:
                        advocate_name = " ".join(name_parts)
                        break
        
        return {
            "query_type": "advocate_search",
            "primary_intent": "find_advocate_cases",
            "advocate_name": advocate_name,
            "requires_deep_search": False,
            "search_terms": [advocate_name] if advocate_name else [],
            "filters": {"advocate_name": advocate_name},
            "response_format": {
                "type": "list",
                "include_metadata": True,
                "include_content": False,
                "group_by": "advocate_role"
            },
            "extracted_entities": {
                "main_subject": f"advocate {advocate_name}" if advocate_name else "advocate",
                "qualifiers": ["cases where advocate appears"],
                "output_fields": ["case_name", "advocate_role", "source_pdf"]
            },
            "confidence": 0.9 if advocate_name else 0.6
        }
    
    def _get_default_classification(self) -> Dict:
        """Return default classification"""
        return {
            "query_type": "explanation_query",
            "primary_intent": "explain_case",
            "requires_deep_search": True,
            "search_terms": [],
            "filters": {},
            "response_format": {
                "type": "explanation",
                "include_metadata": True,
                "include_content": True,
                "group_by": "relevance"
            },
            "extracted_entities": {
                "main_subject": "general legal query",
                "qualifiers": [],
                "output_fields": ["all"]
            },
            "confidence": 0.5
        }

# ─── Enhanced Source Management ─────────────────────────────────────────────────
class SourceManager:
    """Ensures all responses include source PDF information"""
    
    @staticmethod
    def get_source_pdf(metadata: Dict) -> str:
        """Get source PDF name with fallback"""
        return metadata.get('source_pdf', metadata.get('document_name', 'Unknown PDF'))
    
    @staticmethod
    def format_source_section(sources: List[str]) -> str:
        """Format source section consistently"""
        if not sources:
            return "\n**Source:** Unknown PDF"
        
        unique_sources = list(set(sources))  # Remove duplicates
        
        if len(unique_sources) == 1:
            return f"\n**Source:** {unique_sources[0]}"
        else:
            source_section = "\n**Sources:**\n"
            for i, source in enumerate(unique_sources, 1):
                source_section += f"{i}. {source}\n"
            return source_section
    
    @staticmethod
    def format_content_with_source(content: str, source_pdf: str, section: str = "") -> str:
        """Format content with source attribution"""
        if section:
            return f"{content}\n*[Source: {source_pdf} - {section}]*\n"
        else:
            return f"{content}\n*[Source: {source_pdf}]*\n"

# ─── Enhanced Response Generator with Content Analysis ──────────────────────────
class AgenticResponseGenerator:
    def __init__(self):
        self.client = client
        self.source_manager = SourceManager()
    
    def generate_case_explanation(self, relevant_docs: List[Dict], query: str) -> str:
        """Generate comprehensive case explanation from multiple documents"""
        if not relevant_docs:
            return f"No relevant documents found for query: {query}\n\n**Source:** No matching documents"
        
        # Build comprehensive context from all relevant documents
        context_parts = []
        all_sources = []
        
        context_parts.append(f"**Legal Case Analysis Query:** {query}")
        context_parts.append(f"**Number of Relevant Documents:** {len(relevant_docs)}")
        context_parts.append("")
        
        for i, doc_result in enumerate(relevant_docs, 1):
            metadata = doc_result['metadata']
            source_pdf = self.source_manager.get_source_pdf(metadata)
            all_sources.append(source_pdf)
            
            context_parts.append(f"### **Document {i}: {source_pdf}** (Relevance: {doc_result['relevance_score']:.2f})")
            
            # Add metadata information
            context_parts.append(f"**Case Name:** {metadata.get('document_name', 'Unknown')}")
            context_parts.append(f"**Case Type:** {metadata.get('case_type', 'Unknown')}")
            context_parts.append(f"**Court:** {metadata.get('court_name', 'Unknown')}")
            context_parts.append(f"**Judge(s):** {', '.join(metadata.get('judges', []))}")
            context_parts.append(f"**Decision Date:** {metadata.get('decision_date', 'Unknown')}")
            
            # Add parties
            if metadata.get('plaintiffs'):
                context_parts.append(f"**Plaintiffs:** {', '.join(metadata.get('plaintiffs', []))}")
            if metadata.get('defendants'):
                context_parts.append(f"**Defendants:** {', '.join(metadata.get('defendants', []))}")
            if metadata.get('appellants'):
                context_parts.append(f"**Appellants:** {', '.join(metadata.get('appellants', []))}")
            if metadata.get('respondents'):
                context_parts.append(f"**Respondents:** {', '.join(metadata.get('respondents', []))}")
            
            # Add relevant content
            relevant_content = doc_result.get('relevant_content', [])
            if relevant_content:
                context_parts.append(f"**Relevant Content from {source_pdf}:**")
                for content_item in relevant_content:
                    context_parts.append(f"- **{content_item['source_section']}:** {content_item['content']}")
            
            context_parts.append("")
        
        combined_context = "\n".join(context_parts)
        
        # Generate comprehensive explanation using LLM
        system_prompt = """You are an expert legal analyst. Based on the provided legal documents and content, generate a comprehensive explanation of the legal case(s) requested by the user.

Your response should:
1. Provide a clear, comprehensive explanation of the case
2. Include key legal issues, parties involved, and court decisions
3. Reference specific content from the documents provided
4. Maintain professional legal language
5. Cite specific sources throughout your explanation
6. Organize information logically with clear headings
7. Include relevant legal precedents if mentioned
8. Explain the significance and implications of the case

Format your response professionally with clear markdown headings and structure.
Always attribute information to specific source documents."""
        
        user_prompt = f"""Based on the following legal documents and content, provide a comprehensive explanation for this query: "{query}"

Legal Document Information:
{combined_context}

Please provide a detailed, well-structured explanation that covers:
1. Case overview and background
2. Key parties involved
3. Legal issues and arguments
4. Court decisions and reasoning
5. Legal precedents and implications
6. Significance of the case

Make sure to cite the specific source documents throughout your explanation."""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add sources section
            sources_section = self.source_manager.format_source_section(all_sources)
            
            return ai_response + "\n\n" + sources_section
            
        except Exception as e:
            error_response = f"Error generating case explanation: {str(e)}\n\n"
            error_response += "**Available Information:**\n"
            for doc_result in relevant_docs:
                metadata = doc_result['metadata']
                source_pdf = self.source_manager.get_source_pdf(metadata)
                error_response += f"- {metadata.get('document_name', 'Unknown Case')} (Source: {source_pdf})\n"
            
            return error_response + self.source_manager.format_source_section(all_sources)
    
    def generate_advocate_cases_response(self, matching_cases: List[Dict], advocate_name: str) -> str:
        """Generate response for advocate-specific case queries"""
        if not matching_cases:
            return f"No cases found where {advocate_name} is the advocate\n\n**Source:** No matching documents found"
        
        response_parts = []
        all_sources = set()
        
        response_parts.append(f"**Cases where {advocate_name} is the advocate ({len(matching_cases)} cases found):**\n")
        
        for i, case_info in enumerate(matching_cases, 1):
            metadata = case_info['metadata']
            advocate_role = case_info['advocate_role']
            source_pdf = self.source_manager.get_source_pdf(metadata)
            all_sources.add(source_pdf)
            
            # Format role name
            role_display = advocate_role.replace('_', ' ').title()
            
            response_parts.append(f"**{i}. {metadata.get('document_name', 'Unknown Case')}**")
            response_parts.append(f"   **Role:** {role_display}")
            response_parts.append(f"   **Source:** {source_pdf}")
            response_parts.append(f"   **Case Type:** {metadata.get('case_type', 'Unknown')}")
            response_parts.append(f"   **Court:** {metadata.get('court_name', 'Unknown')}")
            response_parts.append(f"   **Judge(s):** {', '.join(metadata.get('judges', []))}")
            response_parts.append(f"   **Decision Date:** {metadata.get('decision_date', 'Unknown')}")
            
            # Show other parties
            if metadata.get('appellants'):
                response_parts.append(f"   **Appellants:** {', '.join(metadata.get('appellants', []))}")
            if metadata.get('respondents'):
                response_parts.append(f"   **Respondents:** {', '.join(metadata.get('respondents', []))}")
            if metadata.get('plaintiffs'):
                response_parts.append(f"   **Plaintiffs:** {', '.join(metadata.get('plaintiffs', []))}")
            if metadata.get('defendants'):
                response_parts.append(f"   **Defendants:** {', '.join(metadata.get('defendants', []))}")
            
            response_parts.append("")
        
        # Add all sources summary
        response_parts.append(self.source_manager.format_source_section(list(all_sources)))
        
        return "\n".join(response_parts)
    
    def generate_filtered_cases_response(self, filtered_cases: List[Dict], filters: Dict, query: str) -> str:
        """Generate response for filtered cases - ALWAYS includes source"""
        if not filtered_cases:
            return f"No cases found matching the criteria: {query}"
        
        response_parts = []
        all_sources = set()
        
        # Add header based on filters
        filter_desc = self._build_filter_description(filters)
        response_parts.append(f"**{filter_desc} ({len(filtered_cases)} cases found):**\n")
        
        for i, case in enumerate(filtered_cases, 1):
            metadata = case.get('metadata', {})
            source_pdf = self.source_manager.get_source_pdf(metadata)
            all_sources.add(source_pdf)
            
            response_parts.append(f"**{i}. {metadata.get('document_name', 'Unknown Case')}**")
            response_parts.append(f"   **Source:** {source_pdf}")
            response_parts.append(f"   **Case Type:** {metadata.get('case_type', 'Unknown')}")
            response_parts.append(f"   **Court:** {metadata.get('court_name', 'Unknown')}")
            response_parts.append(f"   **Judge(s):** {', '.join(metadata.get('judges', []))}")
            response_parts.append(f"   **Decision Date:** {metadata.get('decision_date', 'Unknown')}")
            
            if metadata.get('case_numbers'):
                response_parts.append(f"   **Case Numbers:** {', '.join(metadata.get('case_numbers', []))}")
            
            response_parts.append("")
        
        # Add all sources summary
        response_parts.append(self.source_manager.format_source_section(list(all_sources)))
        
        return "\n".join(response_parts)
    
    def _build_filter_description(self, filters: Dict) -> str:
        """Build description of applied filters"""
        desc_parts = []
        
        if filters.get('case_type'):
            desc_parts.append(f"{filters['case_type']} cases")
        else:
            desc_parts.append("Cases")
        
        if filters.get('judge'):
            desc_parts.append(f"with Judge {filters['judge']}")
        
        if filters.get('court'):
            desc_parts.append(f"in {filters['court']}")
        
        return " ".join(desc_parts)

# ─── Enhanced Legal Data Processor ──────────────────────────────────────────────
class LegalDataProcessor:
    def __init__(self):
        self.client = client
    
    def search_cases_by_advocate(self, advocate_name: str, indices: List[Dict]) -> List[Dict]:
        """Search for cases where a specific advocate is involved"""
        matching_cases = []
        
        # Normalize the advocate name for fuzzy matching
        advocate_name_lower = advocate_name.lower()
        advocate_parts = advocate_name_lower.split()
        
        for idx in indices:
            metadata = idx.get('metadata', {})
            
            # Check all advocate fields
            advocate_fields = [
                'plaintiff_advocates',
                'defendant_advocates', 
                'appellant_advocates',
                'respondent_advocates'
            ]
            
            found_match = False
            advocate_role = None
            
            for field in advocate_fields:
                advocates = metadata.get(field, [])
                for advocate in advocates:
                    if advocate and self._fuzzy_match_advocate_name(advocate_name_lower, advocate.lower()):
                        found_match = True
                        advocate_role = field
                        break
                if found_match:
                    break
            
            if found_match:
                matching_cases.append({
                    'case_data': idx,
                    'advocate_role': advocate_role,
                    'metadata': metadata
                })
        
        return matching_cases
    
    def _fuzzy_match_advocate_name(self, search_name: str, advocate_name: str) -> bool:
        """Fuzzy match advocate names allowing for partial matches"""
        search_parts = search_name.split()
        advocate_parts = advocate_name.split()
        
        # Check if most search terms are found in advocate name
        matches = 0
        for search_part in search_parts:
            if any(search_part in advocate_part for advocate_part in advocate_parts):
                matches += 1
        
        # Return true if at least 70% of search terms match
        return matches >= len(search_parts) * 0.7
    
    def filter_cases_by_criteria(self, indices: List[Dict], filters: Dict) -> List[Dict]:
        """Filter cases based on multiple criteria"""
        filtered_cases = []
        
        for idx in indices:
            metadata = idx.get('metadata', {})
            
            # Apply filters
            if self._matches_filters(metadata, filters):
                filtered_cases.append(idx)
        
        return filtered_cases
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches all filters"""
        # Judge filter
        if filters.get('judge'):
            judges = metadata.get('judges', [])
            judge_match = any(filters['judge'].lower() in judge.lower() 
                            for judge in judges if judge)
            if not judge_match:
                return False
        
        # Case type filter
        if filters.get('case_type'):
            case_type = metadata.get('case_type', '').lower()
            if filters['case_type'].lower() not in case_type:
                return False
        
        # Case name filter
        if filters.get('case_name'):
            doc_name = metadata.get('document_name', '').lower()
            case_name_parts = filters['case_name'].lower().split()
            if not any(part in doc_name for part in case_name_parts):
                return False
        
        # Court filter
        if filters.get('court'):
            court_name = metadata.get('court_name', '').lower()
            if filters['court'].lower() not in court_name:
                return False
        
        return True
    
    def extract_all_advocates(self, metadata: Dict) -> Dict[str, List[str]]:
        """Extract all advocates organized by type"""
        advocates = {
            'plaintiff_advocates': metadata.get('plaintiff_advocates', []),
            'defendant_advocates': metadata.get('defendant_advocates', []),
            'appellant_advocates': metadata.get('appellant_advocates', []),
            'respondent_advocates': metadata.get('respondent_advocates', [])
        }
        
        # Remove empty lists
        advocates = {k: v for k, v in advocates.items() if v}
        return advocates
    
    def extract_all_parties(self, metadata: Dict) -> Dict[str, List[str]]:
        """Extract all parties organized by type"""
        parties = {
            'plaintiffs': metadata.get('plaintiffs', []),
            'defendants': metadata.get('defendants', []),
            'appellants': metadata.get('appellants', []),
            'respondents': metadata.get('respondents', [])
        }
        
        # Remove empty lists
        parties = {k: v for k, v in parties.items() if v}
        return parties

# ─── Enhanced Case Matching Functions ───────────────────────────────────────────
def fuzzy_match_case_name(query: str, document_name: str) -> float:
    """Enhanced case name matching with fuzzy logic"""
    query_words = set(query.lower().split())
    doc_words = set(document_name.lower().split())
    
    stop_words = {'case', 'vs', 'versus', 'ltd', 'inc', 'pvt', 'the', 'and', 'of', 'in', 'v', 'v.', 'who', 'is', 'was', 'were', 'all', 'list', 'advocates', 'parties', 'explain', 'tell', 'me', 'about', 'where', 'advocate'}
    query_words -= stop_words
    doc_words -= stop_words
    
    if not query_words:
        return 0.0
    
    intersection = query_words.intersection(doc_words)
    similarity = len(intersection) / len(query_words)
    
    # Boost for exact matches of key terms
    key_terms = ['western', 'digital', 'burger', 'king', 'nri', 'taxi', 'loreal', 'technologies', 'syt', 'solutions', 'toyota', 'motor', 'corporation', 'ticona', 'polymers']
    for term in key_terms:
        if term in query.lower() and term in document_name.lower():
            similarity += 0.3
    
    return min(similarity, 1.0)

def find_best_matching_case(query: str, indices: List[Dict]) -> Optional[Dict]:
    """Find the best matching case based on query"""
    best_match = None
    best_score = 0.0
    
    for idx in indices:
        metadata = idx.get('metadata', {})
        doc_name = metadata.get('document_name', '')
        
        score = fuzzy_match_case_name(query, doc_name)
        
        # Check case numbers
        case_numbers = metadata.get('case_numbers', [])
        for case_num in case_numbers:
            if any(word in case_num.lower() for word in query.lower().split()):
                score += 0.3
        
        # Check parties
        all_parties = (metadata.get('plaintiffs', []) + 
                      metadata.get('defendants', []) + 
                      metadata.get('appellants', []) + 
                      metadata.get('respondents', []))
        
        for party in all_parties:
            if party and fuzzy_match_case_name(query, party) > 0.3:
                score += 0.2
        
        if score > best_score:
            best_score = score
            best_match = idx
    
    return best_match if best_score > 0.2 else None

def find_all_matching_cases(query: str, indices: List[Dict]) -> List[Dict]:
    """Find all documents matching a case query (handles multiple documents per case)"""
    matching_docs = []
    
    for idx in indices:
        metadata = idx.get('metadata', {})
        doc_name = metadata.get('document_name', '')
        
        score = fuzzy_match_case_name(query, doc_name)
        
        # Check case numbers
        case_numbers = metadata.get('case_numbers', [])
        for case_num in case_numbers:
            if any(word in case_num.lower() for word in query.lower().split()):
                score += 0.3
        
        # Check parties
        all_parties = (metadata.get('plaintiffs', []) + 
                      metadata.get('defendants', []) + 
                      metadata.get('appellants', []) + 
                      metadata.get('respondents', []))
        
        for party in all_parties:
            if party and fuzzy_match_case_name(query, party) > 0.3:
                score += 0.2
        
        if score > 0.2:
            matching_docs.append(idx)
    
    return matching_docs

# ─── Main Enhanced Query Handler ────────────────────────────────────────────────
class AgenticLegalQueryHandler:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.indices = self.load_indices()
        self.classifier = AgenticLegalQueryClassifier()
        self.data_processor = LegalDataProcessor()
        self.response_generator = AgenticResponseGenerator()
        self.content_search = ContentSearchEngine()
        self.source_manager = SourceManager()
    
    def load_indices(self) -> List[Dict]:
        """Load all PageIndex files"""
        indices = []
        for idx_file in glob.glob(os.path.join(self.index_dir, "*_pageindex.json")):
            try:
                with open(idx_file, encoding="utf-8") as f:
                    indices.append(json.load(f))
            except Exception as e:
                print(f"Error loading {idx_file}: {e}")
        return indices
    
    def answer_query(self, query: str) -> str:
        """Answer queries with deep content analysis - ALWAYS includes source information"""
        try:
            # Check if it's an advocate search query first
            if any(keyword in query.lower() for keyword in ["advocate", "harish", "vaidyanathan", "shankar"]):
                advocate_classification = self.classifier.classify_advocate_query(query)
                if advocate_classification.get("advocate_name"):
                    return self.handle_advocate_search(query, advocate_classification["advocate_name"])
            
            # Step 1: Classify the query using agentic AI
            classification = self.classifier.classify_query_with_llm(query)
            
            print(f"Agentic Classification: {classification}")
            
            query_type = classification.get('query_type', 'explanation_query')
            primary_intent = classification.get('primary_intent', 'explain_case')
            requires_deep_search = classification.get('requires_deep_search', False)
            search_terms = classification.get('search_terms', [])
            filters = classification.get('filters', {})
            response_format = classification.get('response_format', {})
            
            # Step 2: Route to appropriate handler based on classification
            if query_type == 'advocate_search':
                advocate_name = filters.get('advocate_name')
                if advocate_name:
                    return self.handle_advocate_search(query, advocate_name)
            
            elif query_type == 'explanation_query' or requires_deep_search:
                return self.handle_explanation_query(query, search_terms)
            
            elif query_type == 'content_analysis':
                return self.handle_content_analysis(query, search_terms)
            
            elif query_type == 'filtered_search':
                return self.handle_filtered_search(query, filters, response_format)
            
            elif query_type == 'comprehensive_list':
                return self.handle_comprehensive_list(query, primary_intent, filters)
            
            elif query_type == 'specific_entity':
                return self.handle_specific_entity(query, primary_intent, filters)
            
            elif query_type == 'complex_filter':
                return self.handle_complex_filter(query, filters, response_format)
            
            else:
                return self.handle_general_query(query)
                
        except Exception as e:
            return f"Error processing query: {str(e)}\n\n**Source:** Error - Unable to determine source"
    
    def handle_advocate_search(self, query: str, advocate_name: str) -> str:
        """Handle advocate-specific search queries"""
        print(f"Searching for cases with advocate: {advocate_name}")
        
        matching_cases = self.data_processor.search_cases_by_advocate(advocate_name, self.indices)
        return self.response_generator.generate_advocate_cases_response(matching_cases, advocate_name)
    
    def handle_explanation_query(self, query: str, search_terms: List[str] = None) -> str:
        """Handle explanation queries with deep content search"""
        print(f"Handling explanation query with deep content search...")
        
        # Perform deep content search across all documents
        relevant_docs = self.content_search.search_content_across_documents(query, self.indices)
        
        if not relevant_docs:
            return f"No relevant documents found for explanation query: {query}\n\n**Source:** No matching documents found"
        
        print(f"Found {len(relevant_docs)} relevant documents")
        
        # Generate comprehensive explanation
        return self.response_generator.generate_case_explanation(relevant_docs, query)
    
    def handle_content_analysis(self, query: str, search_terms: List[str]) -> str:
        """Handle content analysis queries"""
        return self.handle_explanation_query(query, search_terms)
    
    def handle_filtered_search(self, query: str, filters: Dict, response_format: Dict) -> str:
        """Handle filtered search queries"""
        filtered_cases = self.data_processor.filter_cases_by_criteria(self.indices, filters)
        return self.response_generator.generate_filtered_cases_response(filtered_cases, filters, query)
    
    def handle_comprehensive_list(self, query: str, primary_intent: str, filters: Dict) -> str:
        """Handle comprehensive list queries"""
        matching_cases = find_all_matching_cases(query, self.indices)
        
        if not matching_cases:
            return f"No case found matching: {query}\n\n**Source:** No matching documents found"
        
        case_name = matching_cases[0].get('metadata', {}).get('document_name', 'Unknown Case')
        
        # For list queries, use original logic but ensure source inclusion
        if primary_intent == 'list_advocates':
            return self.generate_advocates_response(matching_cases, case_name)
        elif primary_intent == 'get_parties':
            return self.generate_parties_response(matching_cases, case_name)
        else:
            return self.handle_explanation_query(query)
    
    def handle_specific_entity(self, query: str, primary_intent: str, filters: Dict) -> str:
        """Handle specific entity queries"""
        case_match = find_best_matching_case(query, self.indices)
        
        if not case_match:
            return f"No case found matching: {query}\n\n**Source:** No matching documents found"
        
        metadata = case_match.get('metadata', {})
        source_pdf = self.source_manager.get_source_pdf(metadata)
        
        if primary_intent == 'find_judge':
            judges = metadata.get('judges', [])
            if judges:
                return f"**Judge:** {judges[0]}\n**Source:** {source_pdf}"
            else:
                return f"No judge information found for this case\n**Source:** {source_pdf}"
        
        elif primary_intent == 'get_parties':
            party_type = filters.get('party_type', 'plaintiff')
            parties = metadata.get(f"{party_type}s", [])
            if parties:
                return f"**{party_type.title()}:** {', '.join(parties)}\n**Source:** {source_pdf}"
            else:
                return f"No {party_type} information found for this case\n**Source:** {source_pdf}"
        
        else:
            return self.handle_explanation_query(query)
    
    def handle_complex_filter(self, query: str, filters: Dict, response_format: Dict) -> str:
        """Handle complex filter queries"""
        filtered_cases = self.data_processor.filter_cases_by_criteria(self.indices, filters)
        return self.response_generator.generate_filtered_cases_response(filtered_cases, filters, query)
    
    def handle_general_query(self, query: str) -> str:
        """Handle general queries with content search"""
        return self.handle_explanation_query(query)
    
    def generate_advocates_response(self, matching_cases: List[Dict], case_name: str) -> str:
        """Generate advocates response with source attribution"""
        if not matching_cases:
            return f"No advocates found for case: {case_name}\n\n**Source:** No matching documents"
        
        response_parts = []
        all_sources = []
        
        response_parts.append(f"**All Advocates in {case_name}:**\n")
        
        for i, case in enumerate(matching_cases, 1):
            metadata = case.get('metadata', {})
            source_pdf = self.source_manager.get_source_pdf(metadata)
            all_sources.append(source_pdf)
            
            response_parts.append(f"### **Document {i}: {source_pdf}**")
            
            advocates = self.data_processor.extract_all_advocates(metadata)
            
            if not advocates:
                response_parts.append("   *No advocates found in this document*")
                continue
            
            advocate_type_names = {
                'plaintiff_advocates': 'Plaintiff Advocates',
                'defendant_advocates': 'Defendant Advocates',
                'appellant_advocates': 'Appellant Advocates',
                'respondent_advocates': 'Respondent Advocates'
            }
            
            for advocate_type, advocate_list in advocates.items():
                type_name = advocate_type_names.get(advocate_type, advocate_type.replace('_', ' ').title())
                response_parts.append(f"**{type_name}:**")
                for j, advocate in enumerate(advocate_list, 1):
                    response_parts.append(f"   {j}. {advocate}")
                response_parts.append("")
        
        response_parts.append(self.source_manager.format_source_section(all_sources))
        
        return "\n".join(response_parts)
    
    def generate_parties_response(self, matching_cases: List[Dict], case_name: str) -> str:
        """Generate parties response with source attribution"""
        if not matching_cases:
            return f"No parties found for case: {case_name}\n\n**Source:** No matching documents"
        
        response_parts = []
        all_sources = []
        
        response_parts.append(f"**All Parties in {case_name}:**\n")
        
        for i, case in enumerate(matching_cases, 1):
            metadata = case.get('metadata', {})
            source_pdf = self.source_manager.get_source_pdf(metadata)
            all_sources.append(source_pdf)
            
            response_parts.append(f"### **Document {i}: {source_pdf}**")
            
            parties = self.data_processor.extract_all_parties(metadata)
            
            if not parties:
                response_parts.append("   *No parties found in this document*")
                continue
            
            party_type_names = {
                'plaintiffs': 'Plaintiffs',
                'defendants': 'Defendants',
                'appellants': 'Appellants',
                'respondents': 'Respondents'
            }
            
            for party_type, party_list in parties.items():
                type_name = party_type_names.get(party_type, party_type.replace('_', ' ').title())
                response_parts.append(f"**{type_name}:**")
                for j, party in enumerate(party_list, 1):
                    response_parts.append(f"   {j}. {party}")
                response_parts.append("")
        
        response_parts.append(self.source_manager.format_source_section(all_sources))
        
        return "\n".join(response_parts)

# ─── Interactive Query Interface ────────────────────────────────────────────────
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Legal Query Handler with Deep Content Analysis and Advocate Search")
    parser.add_argument("--index_dir", default="pageindices", help="PageIndex directory")
    args = parser.parse_args()
    
    # Initialize query handler
    print("Loading Enhanced Legal Query Handler with Deep Content Analysis and Advocate Search...")
    handler = AgenticLegalQueryHandler(args.index_dir)
    
    print(f"Loaded {len(handler.indices)} legal documents")
   
    print()
    
    # Interactive query loop
    while True:
        query = input("🔍 Enter your legal query (or 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit"):
            break
        
        if query:
            try:
                response = handler.answer_query(query)
                print(f"\n📄 {response}\n")
                print("─" * 80)
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print("**Source:** Error encountered during processing")
        else:
            print("Please enter a valid query.")

if __name__ == "__main__":
    main()
