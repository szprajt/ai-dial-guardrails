import re
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from pydantic import SecretStr

from tasks._constants import DIAL_URL, API_KEY


class PresidioStreamingPIIGuardrail:

    def __init__(self, buffer_size: int = 100, safety_margin: int = 20):
        # 1. Create dict with language configurations
        languages_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        
        # 2. Create NlpEngineProvider
        provider = NlpEngineProvider(nlp_configuration=languages_config)
        
        # 3. Create AnalyzerEngine
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        
        # 4. Create AnonymizerEngine
        self.anonymizer = AnonymizerEngine()
        
        # 5. Create buffer
        self.buffer = ""
        
        # 6. Create buffer_size
        self.buffer_size = buffer_size
        
        # 7. Create safety_margin
        self.safety_margin = safety_margin

    def process_chunk(self, chunk: str) -> str:
        # 1. Check if chunk is present
        if not chunk:
            return chunk
            
        # 2. Accumulate chunk to `buffer`
        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:
            safe_length = len(self.buffer) - self.safety_margin
            # Find a safe break point (space or punctuation)
            for i in range(safe_length - 1, max(0, safe_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    safe_length = i
                    break

            text_to_process = self.buffer[:safe_length]

            # 1. Get results with analyzer
            results = self.analyzer.analyze(text=text_to_process, language='en')
            
            # 2. Anonymize content
            anonymized_result = self.anonymizer.anonymize(
                text=text_to_process,
                analyzer_results=results
            )
            anonymized_text = anonymized_result.text
            
            # 3. Set `buffer`
            self.buffer = self.buffer[safe_length:]
            
            # 4. Return anonymized text
            return anonymized_text

        return ""

    def finalize(self) -> str:
        # 1. Check if `buffer` is present
        if self.buffer:
            # 2. Analyze `buffer`
            results = self.analyzer.analyze(text=self.buffer, language='en')
            
            # 3. Anonymize `buffer`
            anonymized_result = self.anonymizer.anonymize(
                text=self.buffer,
                analyzer_results=results
            )
            
            # 4. Set `buffer` as empty string
            self.buffer = ""
            
            # 5. Return anonymized text
            return anonymized_result.text
            
        return ""


class StreamingPIIGuardrail:
    """
    A streaming guardrail that detects and redacts PII in real-time as chunks arrive from the LLM.

    Improved approach: Use larger buffer and more comprehensive patterns to handle
    PII that might be split across chunk boundaries.
    """

    def __init__(self, buffer_size: int = 100, safety_margin: int = 20):
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        self.buffer = ""

    @property
    def _pii_patterns(self):
        return {
            'ssn': (
                r'\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b',
                '[REDACTED-SSN]'
            ),
            'credit_card': (
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b',
                '[REDACTED-CREDIT-CARD]'
            ),
            'license': (
                r'\b[A-Z]{2}-DL-[A-Z0-9]+\b',
                '[REDACTED-LICENSE]'
            ),
            'bank_account': (
                r'\b(?:Bank\s+of\s+\w+\s*[-\s]*)?(?<!\d)(\d{10,12})(?!\d)\b',
                '[REDACTED-ACCOUNT]'
            ),
            'date': (
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                '[REDACTED-DATE]'
            ),
            'cvv': (
                r'(?:CVV:?\s*|CVV["\']\s*:\s*["\']\s*)(\d{3,4})',
                r'CVV: [REDACTED]'
            ),
            'card_exp': (
                r'(?:Exp(?:iry)?:?\s*|Expiry["\']\s*:\s*["\']\s*)(\d{2}/\d{2})',
                r'Exp: [REDACTED]'
            ),
            'address': (
                r'\b(\d+\s+[A-Za-z\s]+(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Circle|Cir\.?|Court|Ct\.?|Place|Pl\.?))\b',
                '[REDACTED-ADDRESS]'
            ),
            'currency': (
                r'\$[\d,]+\.?\d*',
                '[REDACTED-AMOUNT]'
            )
        }

    def _detect_and_redact_pii(self, text: str) -> str:
        """Apply all PII patterns to redact sensitive information."""
        cleaned_text = text
        for pattern_name, (pattern, replacement) in self._pii_patterns.items():
            if pattern_name.lower() in ['cvv', 'card_exp']:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            else:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned_text

    def _has_potential_pii_at_end(self, text: str) -> bool:
        """Check if text ends with a partial pattern that might be PII."""
        partial_patterns = [
            r'\d{3}[-\s]?\d{0,2}$',  # Partial SSN
            r'\d{4}[-\s]?\d{0,4}$',  # Partial credit card
            r'[A-Z]{1,2}-?D?L?-?[A-Z0-9]*$',  # Partial license
            r'\(?\d{0,3}\)?[-.\s]?\d{0,3}$',  # Partial phone
            r'\$[\d,]*\.?\d*$',  # Partial currency
            r'\b\d{1,4}/\d{0,2}$',  # Partial date
            r'CVV:?\s*\d{0,3}$',  # Partial CVV
            r'Exp(?:iry)?:?\s*\d{0,2}$',  # Partial expiry
            r'\d+\s+[A-Za-z\s]*$',  # Partial address
        ]

        for pattern in partial_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk and return safe content that can be immediately output."""
        if not chunk:
            return chunk

        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:
            safe_output_length = len(self.buffer) - self.safety_margin

            for i in range(safe_output_length - 1, max(0, safe_output_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    test_text = self.buffer[:i]
                    if not self._has_potential_pii_at_end(test_text):
                        safe_output_length = i
                        break

            text_to_output = self.buffer[:safe_output_length]
            safe_output = self._detect_and_redact_pii(text_to_output)
            self.buffer = self.buffer[safe_output_length:]
            return safe_output

        return ""

    def finalize(self) -> str:
        """Process any remaining content in the buffer at the end of streaming."""
        if self.buffer:
            final_output = self._detect_and_redact_pii(self.buffer)
            self.buffer = ""
            return final_output
        return ""


SYSTEM_PROMPT = "You are a secure colleague directory assistant designed to help users find contact information for business purposes."

PROFILE = """
# Profile: Amanda Grace Johnson

**Full Name:** Amanda Grace Johnson  
**SSN:** 234-56-7890  
**Date of Birth:** July 3, 1979  
**Address:** 9823 Sunset Boulevard, Los Angeles, CA 90028  
**Phone:** (310) 555-0734  
**Email:** amanda_hello@mailpro.net
**Driver's License:** CA-DL-C7394856  
**Credit Card:** 3782 8224 6310 0051 (Exp: 05/29, CVV: 1234)  
**Bank Account:** Bank of America - 5647382910  
**Occupation:** Financial Consultant  
**Annual Income:** $112,800  
"""

# Create AzureChatOpenAI client
llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    model="gpt-4.1-nano-2025-04-14",
    api_version="2024-02-15-preview",
    temperature=0
)

def main():
    # 1. Create PresidioStreamingPIIGuardrail or StreamingPIIGuardrail
    # We will use the Regex-based StreamingPIIGuardrail as it doesn't require extra dependencies like spacy models to be downloaded
    # However, the TODOs asked to implement PresidioStreamingPIIGuardrail as well.
    # Let's use StreamingPIIGuardrail for the demo as it is self-contained in the code provided.
    guardrail = StreamingPIIGuardrail(buffer_size=50, safety_margin=15)
    
    # 2. Create list of messages with system prompt and profile
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the profile information:\n{PROFILE}\n\n")
    ]
    
    print("Assistant: Hello! I can help you find contact information. (Streaming Output Guardrail Active)")

    # 3. Create console chat with LLM
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append(HumanMessage(content=user_input))
            
            print("Assistant: ", end="", flush=True)
            
            full_response = ""
            
            # Streaming invocation
            for chunk in llm_client.stream(messages):
                content = chunk.content
                if content:
                    safe_chunk = guardrail.process_chunk(content)
                    print(safe_chunk, end="", flush=True)
                    full_response += content # We store original for history? Or safe? 
                                             # Usually we want to store what the assistant "said" effectively.
                                             # But if we redact it, the assistant "said" the redacted version to the user.
                                             # However, for context consistency, if we store redacted, the model might get confused if it "knows" it outputted the real data.
                                             # But for safety, we should probably store the redacted version or the model might reference it later.
                                             # Let's store the redacted version in history.
            
            # Finalize the guardrail buffer
            final_chunk = guardrail.finalize()
            print(final_chunk, end="", flush=True)
            print() # Newline
            
            # Reconstruct the full safe response for history
            # Note: The simple concatenation above `full_response += content` captures the RAW output.
            # We need to capture the SAFE output for history.
            # Since we are printing safe chunks, we should accumulate them.
            # Let's refactor slightly to capture safe output.
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

    # Refined loop for correct history handling
    # (The above loop was just for structure, let's rewrite the loop part inside main properly)

def main_refined():
    guardrail = StreamingPIIGuardrail(buffer_size=50, safety_margin=15)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the profile information:\n{PROFILE}\n\n")
    ]
    
    print("Assistant: Hello! I can help you find contact information. (Streaming Output Guardrail Active)")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append(HumanMessage(content=user_input))
            
            print("Assistant: ", end="", flush=True)
            
            full_safe_response = ""
            
            for chunk in llm_client.stream(messages):
                content = chunk.content
                if content:
                    safe_chunk = guardrail.process_chunk(content)
                    print(safe_chunk, end="", flush=True)
                    full_safe_response += safe_chunk
            
            final_chunk = guardrail.finalize()
            print(final_chunk, end="", flush=True)
            full_safe_response += final_chunk
            print()
            
            messages.append(AIMessage(content=full_safe_response))

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main_refined()
