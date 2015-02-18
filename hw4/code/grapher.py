from matplotlib import pyplot

s = '''ERROR:hmm:1,LogForm(-4.530880138958478e+18),0.0
ERROR:hmm:2,LogForm(-7.006427858437193e+17),0.0
ERROR:hmm:3,LogForm(-1.647020068696871e+18),0.0
ERROR:hmm:4,LogForm(-1.4572122815379712e+18),0.0
ERROR:hmm:5,LogForm(-1.4593924223029304e+18),0.0
ERROR:hmm:6,LogForm(-1.5826081652551126e+17),0.0
ERROR:hmm:7,LogForm(-5.112813365554429e+17),0.0
ERROR:hmm:8,LogForm(-7.616174413669891e+16),0.0
ERROR:hmm:9,LogForm(-1.8909537684777344e+18),0.0
ERROR:hmm:10,LogForm(-1.6263585978582067e+17),0.0
ERROR:hmm:11,LogForm(-9.553115981454941e+17),0.0'''


short_answers = '''ERROR:hmm:1,LogForm(-4.530880138958478e+18),0.0
ERROR:hmm:2,LogForm(-7.006427858437193e+17),0.0
ERROR:hmm:3,LogForm(-1.647020068696871e+18),0.0
ERROR:hmm:4,LogForm(-1.4572122815379712e+18),0.0
ERROR:hmm:5,LogForm(-1.4593924223029304e+18),0.0
ERROR:hmm:6,LogForm(-1.5826081652551126e+17),0.0
ERROR:hmm:7,LogForm(-5.112813365554429e+17),0.0
ERROR:hmm:8,LogForm(-7.616174413669891e+16),0.0
ERROR:hmm:9,LogForm(-1.8909537684777344e+18),0.0
ERROR:hmm:10,LogForm(-1.6263585978582067e+17),0.0
ERROR:hmm:11,LogForm(-9.553115981454941e+17),0.0
ERROR:hmm:12,LogForm(-3.362245601227622e+17),0.0
ERROR:hmm:13,LogForm(-1.649101089789631e+17),0.0
ERROR:hmm:14,LogForm(-4.032253916567742e+17),0.0
ERROR:hmm:15,LogForm(-2.0791209672885407e+18),0.0
ERROR:hmm:16,LogForm(-1.2162038532441966e+18),0.0
ERROR:hmm:17,LogForm(-2.3533364139528644e+16),0.0
ERROR:hmm:18,LogForm(-1.7445043651683216e+17),0.0
ERROR:hmm:19,LogForm(-6.876388993345262e+17),0.0
ERROR:hmm:20,LogForm(-1.729636935690926e+17),0.0'''

longer_answers = '''ERROR:hmm:1,LogForm(-216879582.8011174),0.0
ERROR:hmm:2,LogForm(-167414912.77789864),0.0
ERROR:hmm:3,LogForm(-121479981.93614615),0.0
ERROR:hmm:4,LogForm(-46931037.972710595),0.0
ERROR:hmm:5,LogForm(-106285360.65889835),0.0
ERROR:hmm:6,LogForm(-118975798.0914532),0.0
ERROR:hmm:7,LogForm(-64313718.68531781),0.0
ERROR:hmm:8,LogForm(-59925905.15469922),0.0
ERROR:hmm:9,LogForm(-67266118.6967948),0.0
ERROR:hmm:10,LogForm(-35451467.3937125),0.0
ERROR:hmm:11,LogForm(-75030748.9731375),0.0
ERROR:hmm:12,LogForm(-59981977.749293506),0.0
ERROR:hmm:13,LogForm(-45906369.841452695),0.0'''

other_answers = '''ERROR:hmm:1,LogForm(-19181.311352709232),0.0
ERROR:hmm:2,LogForm(-23216.975154561165),0.0
ERROR:hmm:3,LogForm(-25577.283437892802),0.0
ERROR:hmm:4,LogForm(-27251.8653699198),0.0
ERROR:hmm:5,LogForm(-28550.713458135724),0.0
ERROR:hmm:6,LogForm(-29611.91225423401),0.0
ERROR:hmm:7,LogForm(-30509.11902850926),0.0
ERROR:hmm:8,LogForm(-31286.298953895002),0.0
ERROR:hmm:9,LogForm(-31971.809274644595),0.0
ERROR:hmm:10,LogForm(-32585.01134828433),0.0
ERROR:hmm:11,LogForm(-33139.71435796927),0.0
ERROR:hmm:12,LogForm(-33646.11418606082),0.0
ERROR:hmm:13,LogForm(-34111.95349944298),0.0
ERROR:hmm:14,LogForm(-34543.250664469466),0.0'''

def parse_input(bigstr):
    lines = bigstr.split('\n')
    ans = []
    for line in lines:
        cutoff = line.split(":")[-1]
        uncomma = cutoff.split(',')[:2]
        print(cutoff, uncomma)
        x = int(uncomma[0])
        y = float(uncomma[1].strip("LogForm()"))
        ans.append((x, y))

    print(ans)

    return ans


def graph(tuples):
    xs, ys = zip(*tuples)
    pyplot.plot(xs, ys)

    pyplot.hold(True)
    pyplot.grid(True)
    pyplot.ylabel("Log likelihood")
    pyplot.xlabel("$N_h$")


    pyplot.show()

# graph(parse_input(short_answers))
graph(parse_input(other_answers))
