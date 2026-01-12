import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import time
import re

INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

PROMPT_TEMPLATE = """You are a handbook assistant. Answer ONLY from the context.
Cite page numbers like "(p. X)". If unsure, say you don't know.

Question: {user_question}

Context:
{context_text}

Answer:"""

@st.cache_resource
def load_models():
    print("Loading models...")
    start_time = time.time()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    db = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    model = genai.GenerativeModel('gemini-pro')
    
    end_time = time.time()
    print(f"Models loaded in {end_time - start_time:.2f} seconds.")
    return db, model

db, model = load_models()

def expand_query(query):
    """Expand query with synonyms for better retrieval"""
    query_lower = query.lower()
    
    # Add related terms based on query
    expansions = []
    
    if 'chapter' in query_lower or 'section' in query_lower:
        expansions.extend(['chapter', 'section', 'structure', 'outline', 'format', 'contents'])
    
    if 'margin' in query_lower:
        expansions.extend(['margin', 'top', 'bottom', 'left', 'right', 'inch', '1.5"', '1.0"', '2.0"', 'formatting requirements'])
    
    if 'spacing' in query_lower:
        expansions.extend(['spacing', 'line', 'paragraph', 'pt', 'point', '1.5 spacing', '6 pt', '6 point'])
    
    if 'development' in query_lower and 'report' in query_lower:
        expansions.extend(['development', 'fyp', 'srs', 'iteration', 'implementation', 'user manual'])
    
    if 'r&d' in query_lower or 'research' in query_lower:
        expansions.extend(['research', 'literature', 'review', 'validation', 'testing', 'results'])
    
    if 'endnote' in query_lower or 'ibid' in query_lower or 'citation' in query_lower:
        expansions.extend(['endnote', 'footnote', 'ibid', 'op. cit', 'citation'])
    
    if 'executive' in query_lower or 'abstract' in query_lower:
        expansions.extend(['executive', 'summary', 'abstract', 'preliminary', 'prelim'])
    
    # Combine original query with expansions
    if expansions:
        return query + " " + " ".join(expansions[:5])
    return query

# Helper function to extract answer from context
def extract_answer_from_context(context, query, retrieved_docs):
    """Extract relevant information directly from retrieved chunks"""
    
    query_lower = query.lower()
    is_formatting_query = any(term in query_lower for term in ['font', 'heading', 'size', 'margin', 'spacing', 'format'])
    
    # Build query-specific keywords
    query_words = [w for w in query.lower().split() if len(w) > 3]
    
    # Add related terms based on query type
    if 'chapter' in query_lower or 'section' in query_lower:
        query_words.extend(['chapter', 'section', 'intro', 'introduction', 'srs', 'requirement', 'implementation', 
                           'manual', 'reference', 'appendix', 'literature', 'review', 'validation', 'testing', 
                           'result', 'discussion', 'conclusion', 'development', 'r&d', 'rd'])
    is_endnote_query = 'endnote' in query_lower or 'footnote' in query_lower or 'ibid' in query_lower or 'op. cit' in query_lower or 'op cit' in query_lower
    if is_endnote_query:
        query_words.extend(['endnote', 'footnote', 'ibid', 'op. cit', 'op cit', 'citation', 'cite', 'reference', 'same source', 'page number'])
    if 'executive summary' in query_lower or 'abstract' in query_lower:
        query_words.extend(['executive', 'summary', 'abstract', 'prelim', 'overview', 'presentation'])
    
    # Collect all relevant sentences with page numbers
    answer_parts = []
    seen_content = set()
    
    margin_info = []
    spacing_info = []
    
    for doc, score in retrieved_docs[:5]:  # Use top 5 (matching retrieval k)
        page = doc.metadata.get('page', 'N/A')
        if page != 'N/A':
            page = int(page) + 1
        
        text = doc.page_content
        
        # Clean up text (remove headers)
        text = re.sub(r'FAST-NUCES\s+\d+', '', text)
        text = re.sub(r'BS Final Year Project Handbook \d+', '', text)
        
        # For formatting queries, also check lines (structured format)
        if is_formatting_query:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) < 5:
                    continue
                
                line_lower = line.lower()
                # Check if line contains formatting specifications
                has_format_spec = (
                    ('font' in line_lower or 'heading' in line_lower or 'size' in line_lower or 
                     'margin' in line_lower or 'spacing' in line_lower) and
                    (re.search(r'=\s*\d+', line) or 'pt' in line_lower or 'arial' in line_lower or 
                     'times' in line_lower or 'roman' in line_lower)
                )
                
                if has_format_spec:
                    if len(line) > 5:
                        line_hash = line.lower()[:60]
                        if line_hash not in seen_content:
                            seen_content.add(line_hash)
                            answer_parts.append(f"{line} (p. {page})")
                    
                    if 'margin' in query_lower:
                        if any(keyword in line_lower for keyword in ['margin', 'top =', 'bottom =', 'left =', 'right =']):
                            margin_info.append((line, page))
                    if 'spacing' in query_lower:
                        if 'spacing' in line_lower and '=' in line_lower:
                            spacing_info.append((line, page))
        
        # For chapter/section queries, look for the format section first
        if 'chapter' in query_lower or 'section' in query_lower:
            # Look for "Development FYP Report Format" or similar section headings
            lines = text.split('\n')
            in_format_section = False
            format_section_lines = []
            found_heading = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if len(line) == 0:
                    continue
                    
                line_lower = line.lower()
                
                # Detect format section heading - be more flexible
                if (('development' in line_lower and ('format' in line_lower or 'report' in line_lower)) or 
                    ('development fyp' in line_lower and 'format' in line_lower) or
                    ('development fyp report format' in line_lower)):
                    in_format_section = True
                    found_heading = True
                    format_section_lines = []
                    continue
                
                # If we're in the format section, collect relevant lines
                if in_format_section:
                    # Collect numbered items (1., 2., etc.) or lines with chapter names
                    is_numbered = re.match(r'^\d+[\.\)]\s+', line)
                    chapter_keywords = ['introduction', 'intro', 'research', 'existing product', 'vision', 
                                       'problem', 'scope', 'stakeholder', 'srs', 'requirement', 'functional', 
                                       'non-functional', 'iteration', 'implementation', 'algorithmic', 
                                       'user manual', 'reference', 'appendix', 'appendices']
                    
                    has_chapter = any(keyword in line_lower for keyword in chapter_keywords)
                    
                    # Stop if we hit a new major section (but allow chapter names)
                    if len(line) > 0 and line[0].isupper() and len(line) < 80:
                        # Check if it's a chapter name or continuation
                        if has_chapter or is_numbered:
                            format_section_lines.append(line)
                        elif len(format_section_lines) > 3:  # We've collected enough, stop
                            break
                    elif has_chapter or is_numbered or (len(line) > 10 and len(line) < 150):
                        # Include if it has chapter keywords, is numbered, or is a reasonable length
                        format_section_lines.append(line)
            
            # If we found the format section, prioritize those lines
            if found_heading and format_section_lines:
                # Extract chapter names from the format section
                chapter_list = []
                for line in format_section_lines[:25]:  # Limit to 25 lines
                    clean_line = line.strip()
                    # Remove numbering (1., 2., etc.)
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    # Extract chapter name (first part before colon, comma, or long description)
                    if ':' in clean_line:
                        clean_line = clean_line.split(':')[0].strip()
                    elif ',' in clean_line and len(clean_line.split(',')[0]) < 50:
                        clean_line = clean_line.split(',')[0].strip()
                    elif len(clean_line) > 80:
                        # If too long, try to extract just the chapter name
                        words = clean_line.split()
                        if len(words) > 0:
                            # Take first few words that look like a chapter name
                            chapter_name = ' '.join(words[:5])
                            if any(kw in chapter_name.lower() for kw in ['introduction', 'research', 'vision', 'srs', 'iteration', 'implementation', 'manual', 'reference', 'appendix']):
                                clean_line = chapter_name
                    
                    if len(clean_line) > 2 and len(clean_line) < 100:
                        line_hash = clean_line.lower()[:60]
                        if line_hash not in seen_content:
                            seen_content.add(line_hash)
                            chapter_list.append(clean_line)
                
                # Add all chapters from format section
                for chapter in chapter_list:
                    answer_parts.append(f"{chapter} (p. {page})")
        
        # Also check for structured lists (lines starting with bullets, numbers, or dashes)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
            
            line_lower = line.lower()
            is_list_item = re.match(r'^[\d\-\•\*]\s+', line) or line.startswith('-') or line.startswith('•')
            
            if 'chapter' in query_lower or 'section' in query_lower:
                # Look for lines that contain chapter/section names
                chapter_keywords = ['introduction', 'intro', 'research', 'existing product', 'vision', 'problem', 
                                   'scope', 'stakeholder', 'srs', 'requirement', 'functional', 'non-functional',
                                   'iteration', 'implementation', 'algorithmic', 'user manual', 'reference', 
                                   'appendix', 'literature review', 'validation', 'testing', 'result', 'discussion',
                                   'conclusion', 'development fyp', 'r&d']
                
                has_chapter_keyword = any(keyword in line_lower for keyword in chapter_keywords)
                
                if has_chapter_keyword and (is_list_item or len(line) < 100):  # Short lines are likely headings
                    clean_line = re.sub(r'^[\d\-\•\*]\s+', '', line).strip()
                    if len(clean_line) > 3:
                        line_hash = clean_line.lower()[:60]
                        if line_hash not in seen_content:
                            seen_content.add(line_hash)
                            answer_parts.append(f"{clean_line} (p. {page})")
            else:
                # For other queries, use original logic
                matches = sum(1 for w in query_words if w in line_lower)
                if (is_list_item and matches >= 1) or matches >= 2:
                    line_hash = line_lower[:60]
                    if line_hash not in seen_content:
                        seen_content.add(line_hash)
                        answer_parts.append(f"{line} (p. {page})")
        
        # Also check regular sentences - but be very strict for chapter queries and endnote queries
        if is_endnote_query:
            # For endnote queries, focus on sentences about Ibid. and op. cit. usage
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                
                sentence_lower = sentence.lower()
                
                # Only extract sentences that mention Ibid. or op. cit. or endnote usage rules
                has_endnote_info = (
                    ('ibid' in sentence_lower and ('same source' in sentence_lower or 'page number' in sentence_lower or 
                     'substitute' in sentence_lower or 'followed by' in sentence_lower or 'repeated' in sentence_lower or
                     'immediate' in sentence_lower)) or 
                    ('op. cit' in sentence_lower or 'op cit' in sentence_lower) or
                    ('endnote' in sentence_lower and ('ibid' in sentence_lower or 'op. cit' in sentence_lower or 
                     'same source' in sentence_lower or 'page number' in sentence_lower))
                )
                
                # Skip if it's about bibliography, references section, or footnotes location
                skip_terms = ['bibliography', 'literature cited', 'references this section', 'footnotes are located',
                             'end notes are like footnotes', 'all of the references', 'list all existing',
                             'references for all works', 'references for quotations', 'footnote comments',
                             'should list all existing', 'information sources used']
                
                if has_endnote_info and not any(term in sentence_lower for term in skip_terms):
                    # Clean up the sentence
                    clean_sentence = sentence
                    # Remove page citations from within the sentence if any
                    clean_sentence = re.sub(r'\s*\(p\.\s*\d+\)\s*', '', clean_sentence)
                    
                    if len(clean_sentence) > 15 and len(clean_sentence) < 300:
                        content_hash = clean_sentence.lower()[:60]
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            answer_parts.append(f"{clean_sentence} (p. {page})")
        elif 'chapter' in query_lower or 'section' in query_lower:
            # For chapter queries, ONLY extract actual chapter names, not descriptions
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 5 or len(sentence) > 200:  # Skip very short or very long sentences
                    continue
                
                sentence_lower = sentence.lower()
                
                # Skip descriptive sentences about chapters
                skip_patterns = [
                    'should contain', 'should give', 'should be', 'will use', 'the reader', 
                    'after reading', 'people reading', 'this section', 'the introduction should',
                    'brief overview', 'background information', 'easy reference', 'important information',
                    'explained in more detail', 'contained in the report'
                ]
                
                if any(pattern in sentence_lower for pattern in skip_patterns):
                    continue
                
                # Only extract if it's clearly a chapter name (short, capitalized, or starts with chapter keyword)
                chapter_keywords = ['introduction', 'intro', 'research', 'existing product', 'vision', 
                                   'srs', 'requirement', 'iteration', 'implementation', 'user manual', 
                                   'reference', 'appendix', 'literature review', 'validation', 'testing']
                
                # Check if sentence starts with a chapter keyword or is a short chapter name
                starts_with_chapter = any(sentence_lower.startswith(kw) or sentence_lower.startswith(kw + ' ') 
                                         for kw in chapter_keywords)
                is_short_chapter_name = len(sentence) < 80 and any(kw in sentence_lower for kw in chapter_keywords)
                
                if starts_with_chapter or (is_short_chapter_name and not any(p in sentence_lower for p in skip_patterns)):
                    # Clean up the sentence to extract just the chapter name
                    clean_sentence = sentence
                    # Remove "The" at the start
                    if clean_sentence.lower().startswith('the '):
                        clean_sentence = clean_sentence[4:].strip()
                    # Take first part if it's too long
                    if len(clean_sentence) > 100:
                        words = clean_sentence.split()
                        # Take first few words that contain chapter keywords
                        for i in range(min(8, len(words))):
                            partial = ' '.join(words[:i+1])
                            if any(kw in partial.lower() for kw in chapter_keywords):
                                clean_sentence = partial
                                break
                    
                    if len(clean_sentence) > 3 and len(clean_sentence) < 150:
                        content_hash = clean_sentence.lower()[:60]
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            answer_parts.append(f"{clean_sentence} (p. {page})")
        else:
            # For non-chapter queries, use original logic
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                
                sentence_lower = sentence.lower()
                
                # Count query word matches
                matches = sum(1 for w in query_words if w in sentence_lower)
                
                # For formatting queries, also check for formatting-related terms
                if is_formatting_query:
                    format_terms = ['font', 'heading', 'size', 'arial', 'times', 'roman', 'bold', 'pt', 'point', 'margin', 'spacing']
                    format_matches = sum(1 for term in format_terms if term in sentence_lower)
                    if format_matches >= 1:  # At least 1 formatting term
                        content_hash = sentence_lower[:50]
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            answer_parts.append(f"{sentence} (p. {page})")
                elif matches >= 1:  # At least 1 query word match (more flexible for general queries)
                    # Filter out very generic sentences
                    irrelevant_terms = ['this document', 'the handbook', 'the reader', 'please note', 
                                        'people reading', 'will use this section', 'should give the reader']
                    
                    if not any(term in sentence_lower for term in irrelevant_terms):
                        content_hash = sentence_lower[:50]
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            answer_parts.append(f"{sentence} (p. {page})")
    
    combined_formatting = []
    if margin_info:
        margin_texts = [re.sub(r'^\s+', '', info[0]).strip() for info in margin_info[:2]]
        combined_formatting.append(f"Margins: {'; '.join(margin_texts)} (p. {margin_info[0][1]})")
    if spacing_info:
        spacing_texts = [re.sub(r'^\s+', '', info[0]).strip() for info in spacing_info[:2]]
        combined_formatting.append(f"Spacing: {'; '.join(spacing_texts)} (p. {spacing_info[0][1]})")
    if combined_formatting:
        answer_parts = combined_formatting + answer_parts
    
    if combined_formatting and ('margin' in query_lower or 'spacing' in query_lower):
        return ". ".join(combined_formatting[:2])
    
    if answer_parts:
        # For endnote queries, filter out irrelevant content
        if is_endnote_query:
            filtered_parts = []
            for part in answer_parts:
                text_part = re.sub(r'\s*\(p\.\s*\d+\)\s*$', '', part).strip()
                text_lower = text_part.lower()
                
                # Skip if it's about bibliography, references section, or general footnotes
                skip_terms = ['bibliography', 'literature cited', 'references this section', 
                             'footnotes are located at the bottom', 'end notes are like footnotes but are located',
                             'all of the references for all works', 'list all existing information sources',
                             'references for quotations', 'footnote comments', 'should list all existing']
                
                if any(term in text_lower for term in skip_terms):
                    continue
                
                # Keep if it mentions Ibid. or op. cit. or endnote usage
                if ('ibid' in text_lower or 'op. cit' in text_lower or 'op cit' in text_lower or
                    ('endnote' in text_lower and ('same source' in text_lower or 'page number' in text_lower or
                     'ibid' in text_lower or 'op. cit' in text_lower))):
                    filtered_parts.append(part)
            
            if filtered_parts:
                return ". ".join(filtered_parts[:5])  # Limit to most relevant
            else:
                return ". ".join(answer_parts[:5])  # Fallback
        
        # For chapter queries, filter out fragments and duplicates more aggressively
        elif 'chapter' in query_lower or 'section' in query_lower:
            # Filter out fragments (very short, incomplete phrases)
            filtered_parts = []
            for part in answer_parts:
                # Remove page citation for checking
                text_part = re.sub(r'\s*\(p\.\s*\d+\)\s*$', '', part).strip()
                
                # Skip if it's a fragment (starts with lowercase, very short, or is clearly incomplete)
                if (len(text_part) < 10 or 
                    text_part.lower().startswith(('reader', 'section', 'behind', 'after', 'the introduction should',
                                                   'it should', 'people', 'will use', 'should give', 'easy reference',
                                                   'important information', 'contained in', 'explained in'))):
                    continue
                
                # Skip if it contains descriptive language
                if any(phrase in text_part.lower() for phrase in [
                    'should contain', 'should give', 'should be', 'will use', 'the reader',
                    'after reading', 'people reading', 'this section', 'brief overview',
                    'background information', 'easy reference', 'important information',
                    'explained in more detail', 'contained in the report', 'needed for the reader'
                ]):
                    continue
                
                # Only keep if it looks like a chapter name (starts with capital, reasonable length)
                if (text_part[0].isupper() if text_part else False) and 10 <= len(text_part) <= 150:
                    filtered_parts.append(part)
            
            if filtered_parts:
                return ". ".join(filtered_parts[:12])  # Allow more items for chapter lists
            else:
                # Fallback: return what we have if filtering removed everything
                return ". ".join(answer_parts[:8])
        
        return ". ".join(answer_parts[:8])  # Max 8 items for comprehensive answers
    return "I do not have that information in the handbook."

# --- 4. Streamlit UI ---
st.title("FAST-NUCES FYP Handbook Assistant")
st.markdown("Ask a question about the BS Final Year Project Handbook 2023.")

# Get user input
query = st.text_input("Your Question:")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the handbook..."):
            
            # 3. Retrieve (Assignment Step 3)
            expanded_query = expand_query(query)
            is_formatting_query = any(term in query.lower() for term in ['margin', 'spacing', 'font', 'heading', 'size', 'format'])
            
            # Use similarity_search_with_relevance_scores to get scores
            # This allows us to check the threshold (Assignment Step 5)
            # Note: FAISS returns similarity scores (higher is better, typically 0-1)
            retrieved_docs = db.similarity_search_with_relevance_scores(
                expanded_query,  # Use expanded query
                k=5  # Increase to 5 for more coverage
            )
            
            # Debug: Show what we retrieved
            if retrieved_docs:
                top_score = retrieved_docs[0][1]
                st.info(f"Top similarity score: {top_score:.3f}")
            
            if not retrieved_docs:
                st.error("I don't have that information in the handbook.")
                st.stop()
            elif retrieved_docs[0][1] < 0.22 and not is_formatting_query:
                st.error("I don't have that information in the handbook.")
                st.stop()
            elif retrieved_docs[0][1] < 0.22 and is_formatting_query:
                st.warning("Low similarity score; attempting to answer with best available formatting info.")
            
               # --- 4. Prepare Context for Prompt ---
            context = ""
            sources = []
            
            for doc, score in retrieved_docs:
                # 'page' metadata comes from the PyMuPDFLoader (8 spaces)
                page = doc.metadata.get('page', 'N/A')
                section_hint = doc.metadata.get('section_hint', 'N/A')
                
                if page != 'N/A': # (8 spaces)
                    # PyMuPDFLoader is 0-indexed, so add 1 for "real" page number (12 spaces)
                    page = int(page) + 1
                
                source_ref = f"Page {page}" # (8 spaces)
                if section_hint != 'N/A' and section_hint != 'Unknown': # (8 spaces)
                    source_ref += f" ({section_hint})" # (12 spaces)
                source_ref += f" (Similarity: {score:.2f})" # (8 spaces)
                
                # Add to context if not already added (to avoid redundant citations) (8 spaces)
                if source_ref not in sources: # (8 spaces)
                    sources.append(source_ref) # (12 spaces)
                    
                context += f"--- Begin Context (Page {page}, Section: {section_hint}) ---\n" # (8 spaces)
                context += doc.page_content # (8 spaces)
                context += f"\n--- End Context (Page {page}) ---\n\n" # (8 spaces)
            
            # --- 5. Generate Answer --- (4 spaces)
            st.subheader("Answer")
            with st.spinner("Generating answer..."):
                final_prompt = PROMPT_TEMPLATE.format(
                    user_question=query,
                    context_text=context
                )
                
                try:
                    response = model.generate_content(final_prompt)
                    llm_answer = response.text.strip()
                    
                    if len(llm_answer) < 20 or '--- Begin Context' in llm_answer or '--- End Context' in llm_answer:
                        answer = extract_answer_from_context(context, query, retrieved_docs)
                    else:
                        answer = llm_answer
                except Exception as e:
                    answer = extract_answer_from_context(context, query, retrieved_docs)
                
                st.markdown(answer)
                
                # 6. Minimal UI (Collapsible sources)
                with st.expander("Sources (Page References)", expanded=True):
                    for source in sources:
                        st.markdown(f"* {source}")
                    st.markdown("---")
                    st.markdown(f"**Retrieved Chunks Debug:**")
                    # Show chunks with metadata
                    debug_info = []
                    for doc, score in retrieved_docs:
                        debug_info.append({
                            "page": doc.metadata.get('page', 'N/A'),
                            "section_hint": doc.metadata.get('section_hint', 'N/A'),
                            "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                            "similarity": f"{score:.3f}",
                            "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        })
                    st.json(debug_info)