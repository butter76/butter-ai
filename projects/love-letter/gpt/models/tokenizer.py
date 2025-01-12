import sys


class LoveLetterTokenizer:
    def __init__(self, debug=False):
        self.debug = debug
        self.special_tokens = {
            'PAD': 0,
            'gamestart': 1,
            'TURNLINE1': 2,
            'TURNLINE2': 3,
            'p1': 4,
            'p2': 5,
            'hidden': 6,
            'draw': 7,
            'play': 8,
            'turn': 9,
            'discard': 10,
            'reveal': 11,
            'lose': 12,
            'end': 13,
            'win': 14,
            'nodraw': 15,
            'highest': 16,
            'NEWLINE': 17,
            'swap': 18,
            'princess': 19,
            'baron': 20,
            'guard': 21,
            'invalid': 22,
            'timeout': 23,
            'yourmove': 24,
            'PLAY1': 25,
            'PLAY2': 26,
            'EOS1': 27,
            'EOS2': 28,
        }
        self.new_line = {
            self.special_tokens['NEWLINE'],
            self.special_tokens['TURNLINE1'],
            self.special_tokens['TURNLINE2'],
            self.special_tokens['PLAY1'],
            self.special_tokens['PLAY2'],
            self.special_tokens['EOS1'],
            self.special_tokens['EOS2']
        }
        
        # Add numbers 1-8 for card values
        for i in range(1, 9):
            self.special_tokens[str(i)] = len(self.special_tokens)
            
        self.vocab_size = len(self.special_tokens)
        # Create reverse mapping for detokenization
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

    def p(self, s):
        if self.debug:
            print(f"[Tokenizer Debug] {s}", file=sys.stderr, flush=True)

    
    def tokenize(self, text: str) -> list[int]:
        self.p(f"Tokenizing text:\n{text}\n-------------------")
        text = text.rstrip('\n')
        tokens = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                self.p(line)
                for word in line.split('|'):
                    word = word.strip()
                    if word:
                        if word in self.special_tokens:
                            tokens.append(self.special_tokens[word])
                            self.p(f"Added token for {word}")
                        else:
                            self.p("AN UNKNOWN WORD: " + word)
                if 'win' in line:
                    if 'p1' in line:
                        tokens.append(self.special_tokens['EOS1'])
                        self.p("Added EOS1 token")
                    else:
                        tokens.append(self.special_tokens['EOS2'])
                        self.p("Added EOS2 token")
                elif 'yourmove' in line:
                    if 'p1' in line:
                        tokens.append(self.special_tokens['TURNLINE1'])
                        self.p("Added TURNLINE1 token")
                    else:
                        tokens.append(self.special_tokens['TURNLINE2'])
                        self.p("Added TURNLINE2 token")
                elif 'play' in line:
                    if 'p1' in line:
                        tokens.append(self.special_tokens['PLAY1'])
                        self.p("Added PLAY1 token")
                    else:
                        tokens.append(self.special_tokens['PLAY2'])
                        self.p("Added PLAY2 token")
                else:
                    tokens.append(self.special_tokens['NEWLINE'])
                    self.p("Added NEWLINE token")
        self.p(f"Final tokens: {tokens}")
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        self.p(f"Detokenizing tokens: {tokens}")
        result = []
        current_line = [""]
        for token in tokens:
            if token in self.id_to_token:
                if token in self.new_line:
                    if current_line:
                        result.append('|'.join(current_line))
                    current_line = [""]
                else:
                    current_line.append(self.id_to_token[token])
        
        if current_line:
            result.append('|'.join(current_line))
        
        final_text = '\n'.join(result)
        self.p(f"Final detokenized text: {final_text}")
        return final_text
