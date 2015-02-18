import random
import difflib
import sys

good = '''When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation. We hold these truths to be self evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security. Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world.'''

bad = '''when in th* course of human*event* it be*omes ne*essar* for one people to d*ssolve the po*itical*bands which have connected the* with another and to assum* among th* powers of the earth the *eparate and equal station to which t*e laws*of nature and of nature s god *ntitle them a decent respect to t*e opinions of*mankind req*ires that they should declare the cau*es which impel them to the *eparation we hold these truths to be self *vident that all men are created e*ual that they are end*wed by their creator wi*h certain *nalienable rights that among these are life li*erty and the p*rsuit of *appiness that to secure these rights governme*ts are instituted among m*n deriving their ju*t powers from the consent of the governed that whenev*r any*form of government becomes des*ructive of these ends it is the right of the people to alter or t* abolish it and to institute new gove*nment laying its *oundation on such principles and orga*izing its powers in*such form as to them shall seem most likely to effect their sa*ety and happiness prudence indeed will*dictate that governments long establ*shed s*ould not be c*anged for li*ht and transient causes and accordingly al* experience hath s*ewn that m*nkind are more disposed to suffer *hile evils*are sufferable than to right*themselv*s by ab*lishing the*forms to whi*h they*are accustomed but *hen a long train of abuses and u*urpations pursu*ng invariabl* the same object*evinces a design to reduce them und*r absolute*despotism it is their right *t is their duty to throw off such *overnment*and to prov*de new guar*s for their f*ture security such has been the*patient sufferance of*these col*nies and such is now the ne*essity which constrains th*m to alter their fo*mer systems of*gover*ment the *istory of the present k*ng of great britai* is a histor* of repeated injuries and usurpations all having in direct*object th* establishment of an*absolute tyra*ny over these states to prove t*is let facts be submitted to a *andid world'''

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def preprocess_good(bad_str):
    to_do = []

    for c in bad_str:
        if c == ' ' or c.lower().islower():
            to_do.append(c.lower())
        elif c == "'":
            to_do.append(' ')
    return "".join(to_do)

def fix(bad, good, alphabet):
    to_emit = []

    print(list(enumerate(bad))[-1])
    print(list(enumerate(bad))[-1])

    for i, c in enumerate(bad):
        if c == '*':
            guess = random.random() < .6
            print('guess was {}'.format(guess))
            to_append = (good[i] if guess else
                         random.choice(alphabet))
        else:
            to_append = c

        to_emit.append(to_append)

    print(''.join(to_emit))

print(len(bad), len(preprocess_good(good)))

prep = preprocess_good(good)

# print(prep)

# diffs = []

# for i, (c1, c2) in enumerate(zip(prep, bad)):
#     if c1 != c2 and c2 != '*':
#         print("diff at {}".format(i))
#         print(c1, c2)

#         print(prep[:i+2])
#         print(bad[:i+2])
#         sys.exit()

# for d in difflib.ndiff(prep, bad):
#     if d[0] != ' ' and d[-1] != '*':
#         print(d)
#         # diffs.append(d)

# print(diffs)


fix(bad, preprocess_good(good), alphabet)
