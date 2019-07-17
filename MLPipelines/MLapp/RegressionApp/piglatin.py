# Authors: CS-World Domination Summer19 - JG
# Function takes in a string, and returns on which has been 
# translated into pig latin
def pig_translate(S):
    retStr = ''
    V = ['a','A','e','E','i','I','o','O','u','U']
    currFront = S[0]
    seen_vowel = False
    for i in range(len(S)):
        if S[i] == ' ':
            if currFront in V:
                retStr += 'yay '
            else:
                retStr += currFront.lower() + 'ay '
        elif i == len(S)-1:
            if currFront in V:
                retStr += S[i]+ 'yay'
            else:
                retStr += S[i] + currFront.lower() + 'ay'
        else:
            if S[i-1] == ' ' or i == 0:
                seen_vowel = False
                currFront = S[i]
                if currFront in V:
                    retStr += S[i]
                    seen_vowel = True
            else:
                if S[i] in V:
                    seen_vowel = True
                    retStr += S[i]
                else:
                    if not seen_vowel:
                        currFront += S[i]
                    else:
                        retStr += S[i]
    return retStr