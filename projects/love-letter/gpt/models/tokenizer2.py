import sys

import torch


class LoveLetterTokenizer:
    def __init__(self, debug=False):
        self.debug = debug
        self.special_tokens = {
            'PAD': 0,
            'gamestart': 1,
            'burn':2,
            'TURNLINE1': 3,
            'TURNLINE2': 4,
            'p1': 5,
            'p2': 6,
            'hidden': 7,
            'draw': 8,
            'play': 9,
            'turn': 10,
            'discard': 11,
            'reveal': 12,
            'lose': 13,
            'end': 14,
            'win': 15,
            'nodraw': 16,
            'highest': 17,
            'NEWLINE': 18,
            'swap': 19,
            'princess': 20,
            'baron': 21,
            'guard': 22,
            'invalid': 23,
            'timeout': 24,
            'yourmove': 25,
            'PLAY1': 26,
            'PLAY2': 27,
            'EOS1': 28,
            'EOS2': 29,
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
        
        #Add values for cards
        for i in range(1, 9):
            self.special_tokens[str(i)] = len(self.special_tokens)

        #Add values for plays
        self.action_tokens = {}
        #guard guesses
        self.action_tokens['1'] = len(self.special_tokens)+len(self.action_tokens)
        for i in range(2, 9):
            self.action_tokens[str('1|' + str(i))] = len(self.special_tokens)+len(self.action_tokens)
        
        #priest thru handmaid
        for i in range(2,5)
            self.action_tokens[str(i)] = len(self.special_tokens)+len(self.action_tokens)
        
        #prince play
        self.action_tokens[str('5|p1')] = len(self.special_tokens)+len(self.action_tokens)
        self.action_tokens[str('5|p2')] = len(self.special_tokens)+len(self.action_tokens)
        
        #other play
        for i in range(6,8)
            self.action_tokens[str(i)] = len(self.special_tokens)+len(self.action_tokens)

        self.vocab_size = len(self.special_tokens)+len(self.action_tokens)
        # Create reverse mapping for detokenization
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        for k,v in self.action_tokens.items():
            self.id_to_token[v] = k

    def p(self, s):
        if self.debug:
            print(f"[Tokenizer Debug] {s}", file=sys.stderr, flush=True)

    
    def pad_tokens(self, tokens, max_length=256):
        pad_length = max_length - len(tokens)
        return tokens[:max_length] + [self.special_tokens['PAD']] * pad_length
    def pad_and_tensor(self, tokens, max_length=256, device='cuda'):
        tokens = self.pad_tokens(tokens, max_length)
        return torch.tensor([tokens], dtype=torch.long).to(device)

    def tokenize(self, text: str) -> list[int]:
        self.p(f"Tokenizing text:\n{text}\n-------------------")
        text = text.rstrip('\n')
        tokens = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                self.p(line)
                action = False
                guard_action = False
                prince_action = False
                for word in line.split('|'):
                    word = word.strip()
                    if not action:
                        if word:
                            if word in self.special_tokens:
                                tokens.append(self.special_tokens[word])
                                self.p(f"Added token for {word}")
                                if word == 'play':
                                    action = True
                            else:
                                self.p("AN UNKNOWN WORD: " + word)
                    else:
                        if guard_action:
                            if not word:
                                tokens.append(self.action_tokens['1'])
                            else:
                                tokens.append(self.action_tokens[str('1|' + word)])
                                guard_action = False
                        elif prince_action:
                            tokens.append(self.action_tokens[str('5|' + word)])
                            prince_action = False
                        else:
                            if word == '1'::
                                guard_action = True
                            elif word == '5':
                                prince_action = True
                            else:
                                tokens.append(self.action_tokens[word])
                                action = False
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
