import os
from gpt.models.tokenizer import LoveLetterTokenizer

def test_tokenizer_roundtrip():
    tokenizer = LoveLetterTokenizer(debug=True)
    logs_dir = "./examples"
    
    for filename in os.listdir(logs_dir):
        if filename.endswith(".log"):
            filepath = os.path.join(logs_dir, filename)
            with open(filepath, 'r') as f:
                original_text = f.read().rstrip('\n')
            lines = original_text.split('\n')
            lines = lines[4:]
            log = '\n'.join(lines)
                
            # Tokenize and then detokenize
            tokens = tokenizer.tokenize(log)
            reconstructed_text = tokenizer.detokenize(tokens).rstrip('\n')
            
            # Compare results
            print(f"Original text:")
            print(log)
            print("-" * 50)
            print(reconstructed_text)
            assert log == reconstructed_text, f"Failed roundtrip test for {filename}"
    print(f"✓ Passed tokenizer test")

if __name__ == "__main__":
    test_tokenizer_roundtrip()