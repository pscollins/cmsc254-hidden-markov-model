import unittest
import hmm
import itertools
import operator

BASE_PATH = '../'
INPUT = '{}/input'.format(BASE_PATH)

gl = None

corrupted = '''when in th* course of human*event* it be*omes ne*essar* for one people to d*ssolve the po*itical*bands which have connected the* with another and to assum* among th* powers of the earth the *eparate and equal station to which t*e laws*of nature and of nature s god *ntitle them a decent respect to t*e opinions of*mankind req*ires that they should declare the cau*es which impel them to the *eparation we hold these truths to be self *vident that all men are created e*ual that they are end*wed by their creator wi*h certain *nalienable rights that among these are life li*erty and the p*rsuit of *appiness that to secure these rights governme*ts are instituted among m*n deriving their ju*t powers from the consent of the governed that whenev*r any*form of government becomes des*ructive of these ends it is the right of the people to alter or t* abolish it and to institute new gove*nment laying its *oundation on such principles and orga*izing its powers in*such form as to them shall seem most likely to effect their sa*ety and happiness prudence indeed will*dictate that governments long establ*shed s*ould not be c*anged for li*ht and transient causes and accordingly al* experience hath s*ewn that m*nkind are more disposed to suffer *hile evils*are sufferable than to right*themselv*s by ab*lishing the*forms to whi*h they*are accustomed but *hen a long train of abuses and u*urpations pursu*ng invariabl* the same object*evinces a design to reduce them und*r absolute*despotism it is their right *t is their duty to throw off such *overnment*and to prov*de new guar*s for their f*ture security such has been the*patient sufferance of*these col*nies and such is now the ne*essity which constrains th*m to alter their fo*mer systems of*gover*ment the *istory of the present k*ng of great britai* is a histor* of repeated injuries and usurpations all having in direct*object th* establishment of an*absolute tyra*ny over these states to prove t*is let facts be submitted to a *andid world'''


@unittest.skip("not worried about LogForm issues")
class TestLogForm(unittest.TestCase):
    # ZERO = hmm.LogForm(0)
    # ONE = hmm.LogForm(1)
    # TWO = hmm.LogForm(2)

    def test_to_float(self):
        for i in range(1, 10):
            self.assertAlmostEqual(float(hmm.LogForm(i)), i)

    def _test_op(self, op):
        for i, j in itertools.product(range(2, 20), range(2, 20)):
            self.assertAlmostEqual(float(op(hmm.LogForm(i), hmm.LogForm(j))),
                                   op(i, j),
                                   msg="{} {} {}".format(i, op, j))
            self.assertAlmostEqual(float(op(hmm.LogForm(i), j)),
                                   op(i, j),
                                   msg="{} {} {}".format(i, op, j))



    def test_add(self):
        self._test_op(operator.add)

    def test_mul(self):
        self._test_op(operator.mul)

    def test_div(self):
        self._test_op(operator.truediv)

    # def test_sub(self):
    #     self._test_op(operator.sub)

# @unittest.skip("not right now")
class TestMarkovModel(unittest.TestCase):
    FILE_PATH_SHORT = "{}/short.txt".format(INPUT)
    FILE_PATH_LONG = "{}/Alice.txt".format(INPUT)

    FILE_PATH = FILE_PATH_SHORT

    N_STATES = 3

    def setUp(self):
        self.model = hmm.MarkovModel(open(self.FILE_PATH),
                                     self.N_STATES)

    @unittest.skip('not worried')
    def test_create(self):
        alphabet = 'abcdefghijklmnopqrstuvqwxyz '
        self.assertSetEqual(set(alphabet), set(self.model.alphabet))

        self.assertEqual(self.model.n_states, self.N_STATES)
        self.assertEqual(len(self.model.initials), self.N_STATES)
        self.assertEqual(len(self.model.transitions[0]), self.N_STATES)
        self.assertEqual(len(self.model.emissions[0]), len(set(alphabet)))

        # global gl
        # gl = self.model

    @unittest.skip('not worried')
    def test_alphas(self):
        alphas = self.model.alphas()

    @unittest.skip('not now')
    def test_betas(self):
        betas = self.model.betas()

    @unittest.skip('no')
    def test_execute_iteration(self):
        self.model._execute_iteration()


    @unittest.skip('no')
    def test_learn(self):
        self.model.learn()

    @unittest.skip("i can't even")
    def test_likelihood(self):
        new_model = hmm.MarkovModel(open(self.FILE_PATH_SHORT),
                                    4,
                                    pseudo_eta=.0001,
                                    pseudo_nu=.0001,
                                    pseudo_mu=.0001)

        hmm.LOG.setLevel(20)

        new_model.learn(30)
        # new_model.generate(150)


   # @unittest.skip("i can't even")
    def test_fix(self):
        new_model = hmm.MarkovModel(open(self.FILE_PATH_SHORT),
                                    2,
                                    pseudo_eta=.0001,
                                    pseudo_nu=.0001,
                                    pseudo_mu=.0001)

        # hmm.LOG.setLevel(20)

        new_model.learn(5)
        new_model.fix(corrupted)

    @unittest.skip("i can't even")
    def test_generate(self):
        new_model = hmm.MarkovModel(open(self.FILE_PATH_LONG),
                                    2,
                                    pseudo_eta=.00000000001,
                                    pseudo_nu=.0000000000001,
                                    pseudo_mu=.0000000000001)

        hmm.LOG.setLevel(50)

        new_model.learn(10)
        new_model.generate(1000000)


    @unittest.skip('not now')
    def test_run(self):
        def make_new_model(x):
            new_model = hmm.MarkovModel(open(self.FILE_PATH_SHORT),
                                        x)
            return new_model

        hmm.LOG.setLevel(40)

        for i in range(1, 21):
            make_new_model(i).learn()

    @unittest.skip('not now')
    def test_big(self):
        def make_new_model(x):
            new_model = hmm.MarkovModel(open(self.FILE_PATH_LONG),
                                        x)
            return new_model

        hmm.LOG.setLevel(40)

        for i in range(1, 21):
            make_new_model(i).learn(5)


    @unittest.skip('not now')
    def test_pseudos(self):
        def make_new_model(x):
            new_model = hmm.MarkovModel(open(self.FILE_PATH_LONG),
                                        x,
                                        pseudo_eta=.0001,
                                        pseudo_nu=.0001,
                                        pseudo_mu=.0001)
            return new_model

        hmm.LOG.setLevel(40)

        for i in range(1, 21):
            make_new_model(i).learn(10)



if __name__ == '__main__':
    unittest.main()
