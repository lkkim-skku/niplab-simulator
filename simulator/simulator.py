
class LearningManager:
    """
    학습에 필요한 자료를 관리합니다.
    fold을 하진 않습니다. folding은 밖에서 합니다.
    """

    def __init__(self):
        self._lcontent, self._ltarget = [], []
        self._econtent, self._etarget = [], []

    def uploadlearn(self, content: list or tuple, target: list or tuple):
        """
        학습 데이터를 업로드합니다.

        :param content: array-like
        :param target: array-like
        :return: None
        """
        self._lcontent, self._ltarget = content, target

    def uploadexam(self, content: list or tuple, target: list or tuple):
        """
        테스트 데이터를 업로드합니다.

        :param content: array-like
        :param target: array-like
        :return:
        """
        self._econtent, self._etarget = content, target

    def learnsource(self):
        """
        학습 데이터를 각 fold마다 yield합니다.

        :yield:
            learning content: array-like \n
            learning target: array-like
        """
        for lc, lt in zip(self._lcontent, self._ltarget):
            yield lc, lt

    def testsource(self):
        """
        테스트 데이터를 각 fold마다 yield합니다.

        :yield:
            examinate content: array-like \n
            examinate target: array-like
       """
        for ec, et in zip(self._econtent, self._etarget):
            yield ec, et

    # def source(self):
    #     """
    #     학습 및 테스트 데이터를 각 fold마다 yield합니다.
    #
    #     :yield:
    #         learning content: array-like \n
    #         learning target: array-like \n
    #         examinate content: array-like \n
    #         examinate target: array-like
    #     """
    #     for lc, lt, ec, et in zip(self._lcontent, self._ltarget, self._econtent, self._etarget):
    #         yield lc, lt, ec, et


class Sim:
    """
    각 classifier를 simulate하고, 그 결과를 정리합니다.
    """
    def __init__(self, simulor, simname: str):
        self._simulor, self._simname = simulor, simname
        self.predlist = []
        self.statistics = {}
        self.pval_list = []
        self.testtarget = []

    @property
    def name(self):
        return self._simname

    @property
    def sim(self):
        return self._simulor

    def simulate(self, fit_data, fit_target, pred_data, pred_target):
        """
        :param fit_data: data for fit
        :param fit_target: target for fit
        :param pred_data: data for predict
        :param pred_target: target for messurement
        :return:self
        fit, predict and measure statistics on each fold
        """

        self._simulor.fit(fit_data, fit_target)
        pred = self._simulor.predict(pred_data)
        self.testtarget.append(pred_target)
        # TODO Reject를 할꺼면 CPON 내부에서 하세요
        # print(len([x for x in pred if 'REJECT' in x]))
        # print(len([x for x, y in zip(pred, pred_target) if 'REJECT' in x and '80' in y]))
        # TODO P-value는 CPON만 쓰니까 metrics_cpon에서 처리하세요
        # if type(self.simulor) == CPON:
        #     self.simulor.pred_pval

        self.predlist.append(pred)


class SimulatorAgent:
    """
    Design Purpose
    --------------
    1. fold learning resource
    2. manage sim
    3. manage learning data and test data
    4. manage test result of simulorules
    """

    def __init__(self, fold=2):
        self.simtaglist = []  # list of classifier
        self._fold = fold  # size of fold
        self._data, self._target = [], []
        self._fold_data, self._fold_target = [], []
        self._learn_data, self._learn_target = [], []
        self._pred_data, self._pred_target = [], []
        self._lm = LearningManager()
        self.unknown = False

    @property
    def folder(self):
        return self._lm

    @folder.setter
    def folder(self, learningmanager):
        self._lm = learningmanager

    def addsim(self, simtag):
        """
        add classifier simulorule with passing by object type check.
        :param simtag:
        :return:
        """
        if isinstance(simtag, Sim):
            self.simtaglist.append(simtag)
        else:
            print('please input SimTag')
        pass

    def fit(self, data, target):
        """
        :param data: {array-like, sparce matrix}, shape = [n_samples, n_features]
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        :param target: array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)
        :return: ClfSim
        """
        lendata, lentarget = len(data), len(target)

        if lendata != lentarget:
            print('Length of X and y are different.')
            return 0
        elif lendata % self._fold != 0:
            print('Length of data is not compitible with fold. modulus is dropped.')

        f = self._fold
        fold_data, fold_target = [[] for _ in range(f)], [[] for _ in range(f)]

        if self.unknown:
            f, self._fold = 6, 6
            # 먼저 반반으로 나눠서 training과 testing이란 이름으로 저장하고
            # 첫번째껀 그냥 저장하고 2번째부터 6번째까지는 5개씩 줄여서 저장
            # 따라서 첫번째는 50개의 class, 6번째는 25개의 class가 저장됨
            # 이젠 여기서 folding을 과정을 모두 처리함
            fold_data, fold_target = [[list(), list()] for _ in range(f)], [[list(), list()] for _ in range(f)]
            targetbuffer = [x for x in set(target)]
            targetbuffer.sort()

            inputdict = {t: [] for t in targetbuffer}
            for d, t in zip(data, target):
                inputdict[t].append(d)

            for i, t in enumerate(targetbuffer):
                lend = len(inputdict[t])
                learndata, testdata = inputdict[t][:int(lend / 2)], inputdict[t][int(lend / 2):]
                # targetindex = int(t)  # 구버전용
                # targetindex = int(t[2:])  # 구버전용(EPxxxx로 naming된 data)
                foldindex_upbound = int(i / 5) + 1
                # index of second-level array: 0 is training, 1 is testing data or target.
                for foldindex in range(6):
                    fold_data[foldindex][1].extend(learndata)
                    fold_target[foldindex][1].extend([t for _ in learndata])
                    if foldindex < foldindex_upbound:
                        fold_data[foldindex][0].extend(learndata)
                        fold_target[foldindex][0].extend([t for _ in learndata])
        else:
            for target_index, zxy in enumerate(list(zip(data, target))):
                foldindex = target_index % f
                fold_data[foldindex].append(zxy[0])
                fold_target[foldindex].append(zxy[1])

        self._fold_data, self._fold_target = fold_data, fold_target
        self._data, self._target = data, target

        return self

    def folding(self):
        fold_data, fold_target, f = self._fold_data, self._fold_target, self._fold

        if self.unknown:
            for j in range(f):
                # index of second-level array: 0 is training, 1 is testing data or target.
                ldata, ltarget = fold_data[j][0], fold_target[j][0]  # learning data
                known_target = set(ltarget)
                pdata, ptarget = fold_data[j][1], [x if x in known_target else 'unknown' for x in fold_target[j][1]]  # predicting data

                self._learn_data, self._learn_target = ldata, ltarget
                self._pred_data, self._pred_target = pdata, ptarget

                yield ldata, ltarget, pdata, ptarget
        else:
            for j in range(f):
                ldata, ltarget = [], []  # learning data
                pdata, ptarget = None, None  # predicting data

                self._learn_data, self._learn_target = ldata, ltarget
                self._pred_data, self._pred_target = pdata, ptarget

                yield ldata, ltarget, pdata, ptarget

    def simulate(self):
        f = 1
        # for fld, flt, fpd, fpt in self.folding():
        for ld, lt, ed, et in self._lm.source():
            for simtag in self.simtaglist:
                simtag.simulate(ld, lt, ed, et)
            print("fold%02d complete" % f)
            f += 1

        # for simtag in self.simtaglist:
        #     ave_stats(simtag)

        return self.simtaglist