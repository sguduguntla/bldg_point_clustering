""" Different String tokenizers for CountVectorizer """

def arka_tokenizer(point):
    """ Tokenizes a string into words based on whenever the character changes from 
    the alphabetic character, to a numberic character, or to any other special character
    that is not alphanumeric.

    :param point: String to be tokenized
    :return: Array of tokens that string is broken up into (array of strings)

    """

    consecutive = dict.fromkeys(["alpha", "numeric", "special char"], False)
    tokens = []
    s = ""
    
    #allowed = ["-", "_"]
    
    for char in point:
        if char.isalpha():
            if not consecutive["alpha"]:
                consecutive["alpha"] = True
                consecutive["numeric"] = False
                consecutive["special char"] = False
                tokens.append(s)
                s = ""
        elif char.isdigit():
            if not consecutive["numeric"]:
                consecutive["alpha"] = False
                consecutive["numeric"] = True
                consecutive["special char"] = False
                tokens.append(s)
                s = ""
        else:                    
            if not consecutive["special char"]:
                consecutive["alpha"] = False
                consecutive["numeric"] = False
                consecutive["special char"] = True
                tokens.append(s)
                s = ""
        
        s += char

    return tokens[1:]