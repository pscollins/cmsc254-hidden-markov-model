
import logging
import math
import numpy
import sys
import itertools

import traceback

logging.basicConfig(level=logging.DEBUG)

LOG = logging.getLogger(__name__)

class LogForm:
    def __init__(self, val=0, already_log=False):

        if not already_log:
            self.logval = math.log(val)
        else:
            self.logval = val

        # LOG.debug('val: {}, self.logval: {}'.format(val, self.logval))

    @staticmethod
    def _get_logval(other):
        # LOG.debug('trying to logval {}'.format(other))
        try:
            other_logval = other.logval
            # LOG.debug('has a logval!')
        except AttributeError:
            # LOG.debug("doesn't have a logval")
            other_logval = math.log(other)


        return other_logval

    def __add__(self, other):
        # LOG.debug('adding self: {}, other: {}'.format(self, other))

        other_logval = self._get_logval(other)

        # LOG.debug('ordering {} and {}'.format(self.logval, other_logval))

        bigger = max(self.logval, other_logval)
        smaller = min(self.logval, other_logval)

        # LOG.debug('bigger: {}, smaller: {}'.format(bigger, smaller))

        new = LogForm(bigger + math.log(1 + math.exp(smaller - bigger)),
                     already_log=True)

        return new

    def __mul__(self, other):
        other_logval = self._get_logval(other)
        new = LogForm(self.logval + other_logval,
                     already_log=True)

        return new

    def __truediv__(self, other):
        other_logval = self._get_logval(other)
        new = LogForm(self.logval - other_logval,
                     already_log=True)

        return new

    def __lt__(self, other):
        other_logval = self._get_logval(other)
        return self.logval < other_logval

    def __float__(self):
        # LOG.debug("i'm being converted to a float!! ({})".format(self))
        # LOG.debug("\n".join(traceback.format_stack()))

        # return math.exp(self.logval)

        return self.logval

    def __repr__(self):
        return 'LogForm({})'.format(self.logval)

    def un_logval(self):
        return math.exp(self.logval)


class MarkovModel:
    RANDOM_DAMPENER = 5
    def __init__(self, file_handle, n_states, missing_symbol='*',
                 pseudo_eta=0, pseudo_mu=0, pseudo_nu=0,
                 max_iter=20):
        self.corpus = [char for line in file_handle if line.strip()
                       for char in line.strip()]
        self.alphabet = tuple(set(self.corpus))
        self.corpus_indicies = self._init_index_array()

        self.pseudo_eta = pseudo_eta
        self.pseudo_mu = pseudo_mu
        self.pseudo_nu = pseudo_nu
        self.max_iter = max_iter
        self.missing_symbol = missing_symbol


        self.n_states = n_states

        LOG.debug('Initialized HMM, len(alphabet): {}, alphabet: {}'.format(
            len(self.alphabet), self.alphabet))

        self.initials = self._init_random_matrix(n_states, 1)[0]
        self.transitions = self._init_random_matrix(n_states, n_states)
        self.emissions = self._init_random_matrix(len(self.alphabet), n_states)

        LOG.info("Initialized random variables")

        LOG.debug("initials: {}".format(self.initials))
        LOG.debug("transitions: {}".format(self.transitions))
        LOG.debug("emissions: {}".format(self.emissions))
        LOG.debug("emissions, as floats: {}".format([[x.un_logval() for x
                                                      in row]
                                                     for row in
                                                     self.emissions]))

        # LOG.debug("sums: {}".format([sum([x.un_logval() for x
        #                                   in row])
        #                                  for row in
        #                                  self.emissions]))

        # LOG.debug("sums: {}".format([sum([x for x
        #                                   in row])
        #                              for row in
        #                              self.emissions]))


        # LOG.debug("corpus: {}".format("".join(self.corpus)))
        # LOG.debug("corpus indicies: {}".format(self.corpus_indicies))

        # self.forwards = None
        # self.backwards = None
        self.gamma_vals = None
        self.alpha_vals = None
        self.beta_vals = None
        self.xi_vals = None

        self.to_map = numpy.array([range(self.n_states)] * len(self.corpus))

        self.to_map_omega = numpy.array([range(len(self.alphabet))] *
                                        self.n_states)

        self.to_map_theta = numpy.array([range(self.n_states)] * self.n_states)


    def learn(self, it_override=None):
        old_likelihood = float('nan')
        LOG.warning("Initial likelihood: {}".format(old_likelihood))

        num_iters = range(it_override if it_override is not None else
                          self.max_iter)

        for i in num_iters:
            #not really a warning but i want a higher level
            LOG.warning("Beginning iteration {}".format(i))
            new_likelihood = self._execute_iteration()

            LOG.warning("New likelihood: {}".format(new_likelihood))

            # if (not math.isnan(old_likelihood) and
            #     (new_likelihood < old_likelihood)):
            #     LOG.warning("Converged! ")
            #     break

            old_likelihood = new_likelihood

        try:
            float_likelihood = new_likelihood.un_logval()
        except Exception:
            float_likelihood = None

        LOG.error("{},{},{}".format(self.n_states, new_likelihood,
                                    float_likelihood))

    def likelihood(self, gammas, xis):
        pi_term = numpy.sum(gammas[0]) * numpy.sum(self.initials)

        LOG.debug("pi term: {}".format(pi_term))

        theta_term = numpy.sum(numpy.sum(self.transitions, axis=0) *
                               numpy.sum(xis, axis=0))

        LOG.debug("theta term: {}".format(theta_term))

        row_sums = []

        for state in range(self.n_states):
            for t, alph in enumerate(self.corpus_indicies):
                 row_sums.append(self.emissions[state][alph] * gammas[t][state])

        omega_term = numpy.sum(row_sums)

        LOG.debug("omega term: {}".format(omega_term))

        total = omega_term + theta_term + pi_term

        LOG.debug("total: {}".format(total))


        return total


    def generate(self, n_chars):
        LOG.info("About to generate a {} character string...".format(n_chars))

        dec_initials = self._logform_convert(self.initials)
        dec_emissions = self._decimalize_arr(self.emissions)
        dec_transitions = self._decimalize_arr(self.transitions)


        LOG.debug("initials (sum={}): {}".format(sum(dec_initials),
                                                 dec_initials))
        LOG.debug("emissions: {}".format(dec_emissions))
        LOG.debug("transitions (sum={}): {}".format(
            numpy.sum(dec_transitions, axis=1),
            dec_transitions))

        current_state = numpy.random.choice(self.n_states,
                                            p=dec_initials)

        LOG.debug("initial state: {}".format(current_state))

        ret = ""

        for _ in range(n_chars):
            # LOG.debug(
            #     'choosing from sum={}, {}'.format(
            #         sum(dec_emissions[current_state]),
            #         dec_emissions[current_state])
            # )


            to_emit = numpy.random.choice(self.alphabet,
                                          p=dec_emissions[current_state])

            LOG.debug("emitting {}".format(to_emit))

            ret += to_emit

            current_state = numpy.random.choice(
                self.n_states,
                p=dec_transitions[current_state])

            LOG.debug("new state: {}".format(current_state))

        print(ret)



    # def fill_missing(self, file_handle):
    #     chars =  [char for line in file_handle if line.strip()
    #               for char in line.strip()]

    #     old_corpus = self.corpus
    #     tmp_corpus = [c for c in chars if c != self.missing_symbol]
    #     self.corpus = tmp_corpus
    #     self.corpus_indicies = self._init_index_array()

    #     alphas = self.alphas()
    #     betas = self.betas()
    #     gammas = self.gammas(alphas, betas)

    #     to_ret = ''




    def _decimalize_arr(self, arr, axis=1):
        return numpy.apply_along_axis(self._logform_convert,
                                      axis,
                                      arr)

    def _logform_convert(self, array_row):
        row_sum = numpy.sum(array_row)
        normalized_row = array_row / row_sum

        return numpy.vectorize(lambda x: x.un_logval())(normalized_row)


    def _execute_iteration(self):
        alphas = self.alphas()
        betas = self.betas()

        gammas = self.gammas(alphas, betas)
        xis = self.xis(gammas, betas)

        self._update_parameters(gammas, xis)

        self.alpha_vals = alphas
        self.beta_vals = betas
        self.gamma_vals = gammas
        self.xi_vals = xis

        return self.likelihood(gammas, xis)

    def _init_index_array(self):
        # return numpy.array([(self.alphabet.index(c) if c in self.alphabet
        #                     else -1) for c
        #                     in self.corpus])
        return numpy.array([self.alphabet.index(c) for c
                            in self.corpus])


    def _update_initials(self, gammas):
        LOG.info("Updating initial probabilities...")

        LOG.debug("gammas[0] = {}".format(gammas[0]))

        def update_initials(x_i):
            numerator = gammas[0][x_i]

            if self.pseudo_eta:
                return ((numerator + self.pseudo_eta) /
                        (1 + self.n_states * self.pseudo_eta))
            else:
                return numerator

        LOG.debug("Old initials: {}".format(self.initials))

        self.initials = numpy.vectorize(update_initials)(self.to_map[0])

        LOG.debug("New initials: {}".format(self.initials))

    def _update_emissions(self, gammas):
        LOG.info("Updating emissions probabilities...")

        def sum_gamma_column(col_index):
            # gs_for_sum = gammas[:-1] if partial else gammas

            def _sum_gamma_column(_yt_filter=None):
                if _yt_filter is not None:
                    # here, we're running over the full set of y_ts, and...
                    filter_array = self.corpus_indicies == _yt_filter

                    # getting an array that we can filter on so we only take
                    # entries from the corpus where we have the letter we care
                    # about

                    gs_for_sum = gammas[filter_array]
                else:
                    gs_for_sum = gammas

                return numpy.sum(gs_for_sum[:, col_index])
            return _sum_gamma_column

        for i, map_row in enumerate(self.to_map_omega):
            full_gamma_sum = sum_gamma_column(i)()

            filtered_gamma_sum = numpy.vectorize(sum_gamma_column(i))(map_row)

            if self.pseudo_mu:
                full_gamma_sum += (self.n_states * len(self.alphabet) *
                                   self.pseudo_mu)
                filtered_gamma_sum += self.pseudo_mu

            self.emissions[i] = filtered_gamma_sum / full_gamma_sum

        # LOG.debug("new emissions: {}".format(self.emissions))


    def _update_transitions(self, gammas, xis):
        LOG.info("Updating transition probabilities...")

        def sum_partial_gammas(col_index):
            return numpy.sum(gammas[:-1][:, col_index])

        def sum_xis(row_index):
            def _sum_xis(col_index):
                # LOG.debug("xis slice: {}".format(xis[:, row_index, col_index]))
                return numpy.sum(xis[:, row_index, col_index])
            return _sum_xis

        for i, map_row in enumerate(self.to_map_theta):
            gamma_sums = sum_partial_gammas(i)
            xi_sums = numpy.vectorize(sum_xis(i))(map_row)

            if self.pseudo_nu:
                gamma_sums += self.n_states ** 2 * self.pseudo_nu
                xi_sums += self.pseudo_nu

            # LOG.debug("updating transisions[{}], mapping on {}".format(
            #     i, map_row
            # ))

            # LOG.debug("gamma_sums: {}".format(gamma_sums))
            # LOG.debug("xi_sums: {}".format(xi_sums))

            # LOG.debug("xis[{}]: {}".format(i, xis[i
                                              # ]))

            self.transitions[i] = xi_sums / gamma_sums

    def _update_parameters(self, gammas, xis):

        # LOG.debug("old params:\n initials = {}\n emissions = {}\n" \
        #           "transitions = {}".format(
        #     self.initials, self.emissions, self.transitions
        # ))

        self._update_initials(gammas)
        self._update_emissions(gammas)
        self._update_transitions(gammas, xis)

        # LOG.debug("new params:\n initials = {}\n emissions = {}\n" \
        #           "transitions = {}".format(
        #     self.initials, self.emissions, self.transitions
        # ))



    def _init_random_matrix(self, width, length, is_logform=True):
        arr = numpy.random.dirichlet(numpy.ones(width) * self.RANDOM_DAMPENER,
                                     size=length)

        if is_logform:
            return numpy.apply_along_axis(numpy.vectorize(LogForm),
                                          0,
                                          arr)
        else:
            return arr


    def _prob_x_given_x(self, x_t2, x_t1):
        return self.transitions[x_t1][x_t2]

    def _prob_y_given_x(self, y_t, x_t):
        return self.emissions[x_t][self.alphabet.index(y_t)]


    def gammas(self, alphas, betas):
        LOG.info("Calculating gammas...")

        # zipped_abs = numpy.dstack((alphas, betas))
        zipped_abs = zip(alphas, betas)

        gammas = numpy.empty([len(self.corpus), self.n_states],
                             dtype=numpy.object)

        # LOG.debug("First row: {}".format(zipped_abs[0]))


        for t, (alpha_row, beta_row) in enumerate(zipped_abs):
            alphas_betas_sum = numpy.sum([alpha_row, beta_row])

            def get_for_x(x_i):
                return (alpha_row[x_i] * beta_row[x_i]) / alphas_betas_sum

            # LOG.debug(
            #     "calculating for t = {}, alpha_row = {}, beta_row = {}".format(
            #         t, alpha_row, beta_row
            #     ))

            gammas[t] = numpy.vectorize(get_for_x)(self.to_map[t])

            # LOG.debug("calculated gammas[{}] = {}".format(t, gammas[t]))


        return gammas


    def xis(self, gammas, betas):
        LOG.info("Calculating xi...")

        x_pairs = list(itertools.product(range(self.n_states),
                                    range(self.n_states)))

        # LOG.debug("x_pairs: {}".format(x_pairs))

        xis = numpy.empty([len(self.corpus) - 1, self.n_states, self.n_states],
                          dtype=numpy.object)

        for t, _ in enumerate(self.corpus[:-1]):
            # LOG.debug("calculating xi for t = {}".format(t))
            for x_old, x_new in x_pairs:
                # LOG.debug("calculating xi for ({}, {})".format(x_old, x_new))
                xis[t][x_old][x_new] = self.calculate_xi(gammas, betas, x_old,
                                                         x_new, t)

            # LOG.debug("xis[{}] = {}".format(t, xis[t]))

        # for row in xis:
        #     for col in row:
        #         for val in col:
        #             assert(val)

        return xis




    def calculate_xi(self, gammas, betas, x_old, x_new, t):
        t_probs = gammas[t][x_old] * betas[t+1][x_new]/betas[t][x_old]
        non_t_probs = (self.transitions[x_old][x_new] *
                       self.emissions[x_new][self.corpus_indicies[t+1]])

        # LOG.debug("t_probs: {}".format(t_probs))
        # LOG.debug("non_t_probs: {}".format(non_t_probs))
        # LOG.debug("t_probs * non_t_probs = {}".format(t_probs * non_t_probs))


        assert(t_probs * non_t_probs)

        return t_probs * non_t_probs


    def alphas(self):
        LOG.info("Calculating forward probabilities...")

        def initialize(x_i):
            # LOG.debug("em[{}][0]: {}".format(
            #     x_i, self.emissions[x_i][self.corpus_indicies[0]]))
            return (self.emissions[x_i][self.corpus_indicies[0]] *
                    self.initials[x_i])

        def grab_pxs(x_t):
            # LOG.debug("grabbing pxs: {}".format(self.transitions[:,x_t]))

            return self.transitions[:,x_t]

        def for_t(y_t):
            def over_xt(x_t):
                # LOG.debug("sum of pxs: {}".format(numpy.sum(grab_pxs(x_t))))
                return self.emissions[x_t][y_t] * numpy.sum(grab_pxs(x_t))
            return over_xt


        alphas = numpy.empty([len(self.corpus), self.n_states],
                             dtype=numpy.object)

        to_map = numpy.array([range(self.n_states)] * len(self.corpus))

        # LOG.debug("About to set alphas[0] (now {})".format(alphas[0]))

        # LOG.debug("emissions[0:2] = {}".format(self.emissions[0:2]))
        # LOG.debug("corpus_indicies[0] = {}".format(self.corpus_indicies[0]))
        # LOG.debug("initials[0:2] = {}".format(self.emissions[:2]))
        # LOG.debug("About to set alphas[0] (now {})".format(alphas[0]))

        alphas[0] = numpy.vectorize(initialize)(to_map[0])

        LOG.debug("Initialized alphas[0] to {}".format(alphas[0]))


        t = None
        for t, y_t in enumerate(self.corpus_indicies[1:], start=1):
            # LOG.debug("Calculating alphas for t = {}, y_t = {}".format(t, y_t))

            prev_alphas = numpy.sum(alphas[t-1])
            # if t > 15:
            #     sys.exit()

            # LOG.debug("prev_alphas: {}".format(prev_alphas))

            alphas[t] = numpy.vectorize(for_t(y_t))(to_map[t]) * prev_alphas

            # LOG.debug("alphas[t] = {}".format(alphas[t]))
            # assert(not math.isnan(alphas[t][0]))

        # LOG.debug("Final alphas: {}".format(alphas[t]))

        # alphas_norm = numpy.empty([self.n_states])
        # final_alphas_sum = numpy.sum(alphas[t])

        # alphas_norm = numpy.vectorize(lambda x: x/final_alphas_sum)(alphas[t])

        # LOG.debug("Normalized alphas: {}".format(alphas_norm))

        return alphas


    def betas(self):
        LOG.info("Calculating backwards probabilities...")
        end = len(self.corpus) - 1

        to_map = numpy.array([range(self.n_states)] * len(self.corpus))
        betas = numpy.empty([len(self.corpus), self.n_states],
                             dtype=numpy.object)


        def grab_pxs(x_t):
            return self.transitions[:,x_t]

        def for_t(y_t):
            # LOG.debug("grabbing for y_t = {}".format(y_t))
            def over_xt(x_t):
                # LOG.debug("pxs: {}".format(grab_pxs(x_t)))
                return (numpy.sum(self.emissions[x_t][y_t]) *
                        numpy.sum(grab_pxs(x_t)))
            return over_xt


        LOG.debug("setting betas[end]")
        betas[end] = numpy.vectorize(lambda x: LogForm(1))(to_map[end])
        LOG.debug("betas[end] = {}".format(betas[end]))

        ts_and_yts = reversed(list(enumerate(self.corpus_indicies))[1:])
        t = None

        for t, y_t in ts_and_yts:
            # LOG.debug("calculating betas for t+1 = {}, y_t+1 = {}".format(t,
            #                                                               y_t))
            # LOG.debug("previous row: betas[{}]={}".format(t, betas[t]))

            prev_betas = numpy.sum(betas[t])
            # LOG.debug("prev_betas = {}".format(prev_betas))

            betas[t-1] = numpy.vectorize(for_t(y_t))(to_map[t]) * prev_betas

            # LOG.debug("Set betas[{}] = {}".format(t, betas[t]))


        # LOG.debug("Final betas: {}".format(betas[t]))

        return betas




    def fix(self, to_fix):
        to_emit = ''

        star_indicies = [i for i, c in enumerate(to_fix) if c == '*']

        for t, character in enumerate(to_fix):
            if character == '*':
                max_state = numpy.argmax(self.gamma_vals[t % len(self.corpus)])
                max_char = numpy.argmax(self.emissions[max_state])
                LOG.debug('index of max char {}'.format(max_char))
                to_emit += self.alphabet[max_char]
            else:
                to_emit += character
        print(to_emit)

    def fix(to_fix):
        table = numpy.zeros([self.n_states, len(to_fix)])
        paths = numpy.zeros([self.n_states, len(to_fix)])

        table[:,0] = self.initials * self.emissions[:, to_fix[0]]

        fixed = [c for c in to_fix]

        for t, character in enumerate(to_fix):
            if character == '*':


            for state in range(self.n_states):
                last_vals = table[:, t-1]
                transitions = self.transitions[:, state]
                emissions = self.emissions[:, self.alphabet.index(character)]
                vals = last_vals * transitions * emissions

                table[state, t] = max(vals)
                path[state, t] = numpy.argmax(vals)
