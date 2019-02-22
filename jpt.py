import numpy as np
import xlrd
import re
from itertools import *
from  math import factorial
import xlsxwriter
import operator
import load_jpt_data as jpt
from itertools import *
import xlrd
import re
import itertools
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier





class jpt:

    directory = '/Users/ashkanekhtiari/Documents/jpt/'
    jgfile = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/jgasht1.npy'
    weight5file = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/weights5.npy'
    weight2file = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/weights2.npy'
    deep5file = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/deeps5.npy'
    deep2file = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/deeps2.npy'
    classfile = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/classdict.npy'
    totalpermutations = 95344200   #this is the same as len(self.jg)
    probabilityrow = 0.00000001    #this is the same as 1/95344300

    tblfile = 'jackpot-table-correct.xlsx'
    t = ['y', 'd', 'b', 's', 'c', 'p']

    def __init__(self):
        self.loadData()

    def randomsizedraw(self):
        num = int(536000000 / (len(self.X) - 133))
        return num - (num % 10 ** (len(str(num)) - 1))


    def jgdata(self):
        indx = np.arange(len(self.jg))
        import time
        tstime = time.time()
        for i in range(133,len(self.X1)):
            print('processing row number %1d starts' % i)
#            randindex = np.sort(np.random.randint(len(indx) , size=1000000), kind='mergesort')[::-1]
            randindex = np.random.randint(len(indx) , size=10000)
            r0 = np.array(self.X1[i],dtype=int)
            klas = self.cls[tuple(self.clas(r0))]
            r0= np.append(r0 ,[[self.calculateClauseValues(self.X1[i] , i)]])
            r0= np.append(r0 , [[int(klas[1]*10**8),int(klas[2]*10**8),klas[4],klas[5] , 52]])
            if 'data' not in dir():
                data = np.array(r0)
            else:
                data = np.vstack((data,r0))
            print("data array has been created")
            stime = time.time()
            for r in randindex:
                r1 = np.array(self.jg[r],dtype=int)
                klas = self.cls[tuple(self.clas(r1))]
                c = self.clausescore(r1,self.X1[i])
                r1 = np.append(r1 , [self.calculateClauseValues(np.array(r1) , i)])
                r1 = np.append(r1 , [[int(klas[1]*pow(10,8)),int(klas[2]*pow(10,8)),klas[4],klas[5] , c ]])
                data = np.vstack((data,r1))
            etime = time.time()
            print("time collapsed for row %1d  = %1d " % (i, (etime - stime)))
            print('processing row number %1d completed!' % i)

        self.MLdata = data
        np.save('datatable.npy' , data)
        return

    def jgdatacomp(self,index):
        data = np.load('datatable.npy')
        print("len data = " , len(data))
        indx = np.arange(len(self.jg))
        import time
        tstime = time.time()
        for i in range(index, len(self.X1)+1):
            print('processing row number %1d starts' % i)
            #            randindex = np.sort(np.random.randint(len(indx) , size=1000000), kind='mergesort')[::-1]
            randindex = np.random.randint(len(indx), size=10000)
            r0 = np.array(self.X1[i], dtype=int)
            klas = self.cls[tuple(self.clas(r0))]
            r0 = np.append(r0, [[self.calculateClauseValues(self.X1[i], i)]])
            r0 = np.append(r0, [[int(klas[1] * 10 ** 8), int(klas[2] * 10 ** 8), klas[4], klas[5], 52]])
            data = np.vstack((data, r0))
            print("row has been added to data ", r0)
            print("len data = " , len(data))

            stime = time.time()
            for r in randindex:
                r1 = np.array(self.jg[r], dtype=int)
                klas = self.cls[tuple(self.clas(r1))]
                c = self.clausescore(r1, self.X1[i])
                r1 = np.append(r1, [self.calculateClauseValues(np.array(r1), i)])
                r1 = np.append(r1, [[int(klas[1] * pow(10, 8)), int(klas[2] * pow(10, 8)), klas[4], klas[5], c]])
                data = np.vstack((data, r1))
            etime = time.time()
            print("time collapsed for row %1d  = %1d " % (i, (etime - stime)))
            print('processing row number %1d completed!' % i)
            print("len data = " , len(data))

        self.MLdata = data
        np.save('datatable1.npy', data)
        return

    def loadData(self):
        # loading all happened JP clauses until now
        self.loadJP()
        # loading all possible clauses
        self.loadJg()
        # loading all weights
        self.loadWeights()
        self.loadDeeps()
        self.loadClass()
        self.statisticArray()
        # calculating last w , d and wd values
#        self.calculateValues(len(self.w5) -1)

    # loads JP table from excel file
    def loadCLS(self):
        file = '/Users/ashkanekhtiari/PycharmProjects/jpt/venv/classification_table_data_final.xlsx'
        wb = xlrd.open_workbook(filename=file)
        ws = wb.sheet_by_name('classification_table_data_print')
        cd = dict()

        cd = dict()
        for r in range(0, ws.nrows):
            k = []
            for row in range(1, 8):
                k.append(ws.cell(r, row).value)
            k = tuple(k)
            v = [int(ws.cell(r, 8).value), ws.cell(r, 9).value, ws.cell(r, 16).value, ws.cell(r, 11).value,
                 int(ws.cell(r, 12).value), int(ws.cell(r, 13).value), int(ws.cell(r, 0).value)]
            cd[k] = v

        np.save("classdict.npy" , cd)
        self.cls = cd
        return

    #loads JP table from excel file
    def loadJP(self):
        file = self.directory + self.tblfile
        wb = xlrd.open_workbook(filename=file)
        ws = wb.sheet_by_name('Sheet1')

        jp = np.zeros((ws.nrows - 1, 33), dtype=int)
        for r in range(1, ws.nrows):
            jp[r - 1, 0] = int(ws.cell(r, 1).value)
            jp[r - 1, 1] = int(re.sub("[^0-9]", "", ws.cell(r, 2).value))
            for c in range(3, 34):
                jp[r - 1, c - 1] = int(ws.cell(r, c).value)

        self.data = jp
        self.X = self.data[:,:9]
        self.Y = self.data[:,9:]
        self.X1 = self.X[:,2:]
        self.y1 = self.Y[:,-2]
        self.y2 = self.Y[:,-1]
        self.pnj = self.X1[:,:5]
        self.do = self.X1[:,5:]

        return

# loads Classification table from excel file
    def loadClass(self):
        self.cls = np.load(self.classfile).item()
        return


    # loads all jaygasht from file 90M
    def loadJg(self):
        self.jg = np.load(self.jgfile)

#loads all the weigth tables for each row in jp
    def loadWeights(self):
        self.w5 = np.load(self.weight5file).tolist()
        self.w2 = np.load(self.weight2file).tolist()
        diff = len(self.X) - len(self.w5)
        if diff > 0:
            self.updateWeights()

#loads all deep table for each row in jp
    def loadDeeps(self):
        self.d5 = np.load(self.deep5file).tolist()
        self.d2 = np.load(self.deep2file).tolist()
        diff = len(self.X) - len(self.d5)
        if diff > 0:
            self.updateDeeps()

    def updateWeights(self):
        self.countWeights()
        self.loadWeights()


    def countWeights(self):
        panjcombs = dict()
        docombs = dict()
        self.w5 = []
        self.w2 = []

        for row in self.X1:
            cmb = self.allcombs(row[:5])
            for x in cmb:
                if x in panjcombs.keys():
                    panjcombs[x] += 1
                else:
                    panjcombs[x] = 1
            w = dict()
            for k, v in panjcombs.items():
                w[k] = v
            self.w5.append(w)

        for row in self.X1:
            cmb = self.allcombs(row[5:])
            for x in cmb:
                if x in docombs.keys():
                    docombs[x] += 1
                else:
                    docombs[x] = 1
            w = dict()
            for k, v in docombs.items():
                w[k] = v
            self.w2.append(w)

        self.wn5 = dict()
        self.wn2 = dict()

        for i in range(1,51):
            self.wn5[i] = self.w5[i]
        for i in range(1,11):
            self.wn2[i] = self.w2[i]

        np.save('weights5.npy' , self.w5)
        np.save('weights2.npy' , self.w2)

        return


    def updateDeeps(self):
        self.countDeeps()
        self.loadDeeps()

    def countDeeps(self):

        deep = dict()
        deep2 = dict()
        for i in range(1, 51):
            deep[i] = 0

        for i in range(1, 11):
            deep2[i] = 0

        self.d5 = []
        self.d2 = []

        for row in self.X1:
            r5 = row[0:5]
            r2 = row[5:]
            for r in r5:
                deep[r] = deep[r] + i + 1
            for r in r2:
                deep2[r] = deep2[r] + i + 1

            d = dict()
            for k, v in deep.items():
                d[k] = v
            self.d5.append(d)

            d = dict()
            for k,v in deep2.items():
                d[k] = v
            self.d2.append(d)

        np.save('deeps5.npy' , self.d5)
        np.save('deeps2.npy' , self.d2)

        return

    def loadWeightTable(self):
        wt = [0 for i in range(len(self.X1)) ]
        return

    def addValue(self , val , res , no , n):
#       print("Value : %d ,  no: %d  , n : %d " % (val, no , n))
        try:
            val += res[no][n]
        except KeyError:
            val += 0
        return val

    def clausescore(self,c,ref):
        s5=0
        s2=0
        for i in c[:5]:
            if i in ref[:5]:
                s5 += 1
        for i in c[5:]:
            if i in ref[5:]:
                s2 += 1
        res = (s5*10+s2)
        return (res if (res > 2) else 0)

    def scoredraw(self,drawno):
        ref = self.X1[drawno]
        return [x + [self.clausescore(x,ref)] for x in self.jg]

    def plotclausewin(self):
        isin = [self.clausescore(self.X1[i], self.X1[i + 1]) for i in range(len(self.X1) - 1)]
        plt.plot(isin)
        return

    def plotoccures(self):
        pi = [self.w5[len(self.X1) - 1][p] for p in range(1, 51)]
        pin = dict()
        for i in range(1, 51):
            pin[i] = pi[i - 1]
        pi5 = OrderedDict(sorted(pin.items(), key=operator.itemgetter(1)))
        pi = [self.w2[len(self.X1) - 1][p] for p in range(1, 11)]
        pin = dict()
        for i in range(1, 11):
            pin[i] = pi[i - 1]
        pi2 = OrderedDict(sorted(pin.items(), key=operator.itemgetter(1)))
        plt.subplot(211)
        plt.bar(range(1,51), pi5.values())
        plt.xticks(range(1,51),pi5.keys())
        plt.yticks(list(pi5.values()) , pi5.values())
        plt.subplot(212)
        plt.bar(range(1,11), pi2.values())
        plt.xticks(range(1,11),pi2.keys())
        plt.yticks(list(pi2.values()) , pi2.values())
        return


    def getClauseWeight(self , r , no):
        pnj = c[:5]
        do = c[5:]
        wval = 0
        dval = 0
        cmb5 = self.allcombs(pnj)
        cmb2 = self.allcombs(do)
        for i5 in cmb5:
            wval += (self.w5[no])[i5]
            dval += (self.d5[no])[i5]
        for i2 in cmb2:
            wval += (self.w5[no])[i2]
            dval += (self.d5[no])[i2]
        return [wval , dval]


    def calculateClauseValues(self , c , no ):
        pnj = c[:5]
        do = c[5:]
        wval = 0
        dval = 0
        for n in pnj:
            wval =  self.addValue(wval ,  self.w5 ,no ,n)
            dval =  self.addValue( dval ,  self.d5,no,n)
        for n in do:
            wval = self.addValue(wval ,self.w2,no,n)
            dval = self.addValue(dval ,self.d2,no,n)

        return [wval , dval]

    def calculateValues(self ,  no):
        self.wval = []
        self.dval = []
        self.wdval = []
        for row in self.X1:
            wval , dval = self.calculateClauseValues(row , no)
            self.wval.append(wval)
            self.dval.append(dval)
            self.wdval.append(wval + dval)

        return

    def appendToDic(self , dic , val , ar):
        try:
            if val in dic.keys():
                e = dic[val]
            else:
                e = []

        except KeyError:
            e = []

        dic[val] = e.append(ar)
        return


    def calculateValueTable(self):
        self.valueTable = []
        wdict = dict()
        ddict = dict()
        wddict = dict()
        maxw = 0
        maxd = 0
        for i in range(0,len(self.X1)):
            starttime = time.time()
            print( i , "-starting to process:  ", self.X1[i])
            for r in self.jg:
                wval , dval = self.calculateClauseValues(r , i)
                print("processing " + r + "  Weigth value : %f  deep value : %f " % (wval , dval))
                self.appendToDic(wdict ,wval, [r,wval])
                self.appendToDic(ddict ,dval,[r,dval])
                self.appendToDic(wddict,wval+dval,[r,wval+dval])
                if wval > maxw:
                    maxw = wval
                if dval > maxd :
                    maxd = dval

            self.valueTable.append([i , self.X1[i],wdict , ddict, wddict , maxw , maxd])
            print(i, maxw, maxd, self.calculateClauseValues(self.X1[i] , i))
            endtime = time.time()
            print("Elapsed time for processing row no " , i , " : " , (endtime-starttime))

        np.save("valueTable.npy" , self.valueTable)

        return

    def statisticArray(self):
 #       self.X2 = [list(x) +[self.calculateClauseValues(x , self.X1.index(x))] +[self.cls[tuple(self.clas(x))][-1]] for x in self.X1]
 #       self.X2 = [list(x) + [self.cls[tuple(self.clas(x))][-1]] for x in self.X1]
        self.X2 = [list(self.X1[i]) + self.calculateClauseValues(self.X1[i], i) + [self.cls[tuple(self.clas(self.X1[i]))][-1]] for i in range(len(self.X1))]
        self.X3 = [list(x)  + [x[7]+x[8]+x[9]] for x in self.X2]
        return

    def KNNeighbor(self):
        X = [s[7:10] for s in self.X2]
        y = [[52] for s in self.X2]
        for r in range(1, 12):
            randindex = np.random.randint(95000000, size=1000000)
            X = X + [self.calculateClauseValues(self.jg[i], 310 + r) + [self.cls[tuple(self.clas(self.jg[i]))][-1]] for i in randindex]
            y = y + [self.clausescore(self.jg[i], self.X1[310 + r]) for i in randindex]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for n_neighbors, ax in zip([1, 3, 9], axes):
            clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
            print(clf.predict(X_test))
            print(clf.score(X_test, y_test))

        return

    def createXyClassification(self):
        self.Xtrain = [s+[52] for s in self.X2]
        X2 = [list(j.jg[i]) + j.calculateClauseValues(j.jg[i], 321) + [j.cls[tuple(j.clas(j.jg[i]))][-1]] + [
            clausescore(j.jg[i], j.X1[321])] for i in ranindex]
        randindex = np.random.randint(95000000, size=10000000)

        # to be reviewed
        X2 = [s + [52] for s in j.X2]
        for r in range(1, 12):
            randindex = np.random.randint(95000000, size=1000000)
            X2 = X2 + [
                list(j.jg[i]) + j.calculateClauseValues(j.jg[i], 310 + r) + [j.cls[tuple(j.clas(j.jg[i]))][-1]] + [
                    j.clausescore(j.jg[i], j.X1[310 + r])] for i in randindex]

            X = [ s[:10] for s in X2 ]
            Y = [s[10] for s in X2 ]
            X1 = [ s[7:10] for s in X2 ]

        return
    #TO DO:
# Update ValueTable

    def updateValueTable(self):



        return


# calculate winner table
    def calculateWinnerTable(self):
        self.winnerTable = []
        wdict = dict()
        ddict = dict()
        wddict = dict()
        maxw = 0
        maxd = 0
        for i in range(0,len(self.X1)):
            for r in self.jg:
                wval , dval = self.calculateClauseValues(r , i)
                self.appendToDic(wdict ,wval, [r,wval])
                self.appendToDic(ddict ,dval,[r,dval])
                self.appendToDic(wddict,wval+dval,[r,wval+dval])
                if wval > maxw:
                    maxw = wval
                if dval > maxd :
                    maxd = dval

            self.valueTable.append([i , self.X1[i],wdict , ddict, wddict , maxw , maxd])

        np.save("valueTable.npy" , self.valueTable)

        return

    def printData(self):
        for r in self.data:
            print(r)

        return

    def printX(self):
        print(self.X)


    def clas(self,r):
        res = []
        for i in r:
            res.append(self.t[int(i/10)])
        return res


    def allcombs(self ,ar):
        per = []
        for row in ar:
            per = per + [row]
        per = per + list(itertools.combinations(ar, 2))
        per = per + list(itertools.combinations(ar, 3))
        per = per + list(itertools.combinations(ar, 4))
        return per

    def plotDeeps(self):
        plt.figure(1)
        plt.subplot(211)
        for i in range(1, 51):
            plt.bar(i, self.d5[i], color="red")
            plt.bar(i, self.pnj[i] * 100, color="blue")
        plt.subplot(212)
        for i in range(1,11):
            plt.bar(i,self.d2[i] , color="red")
            plt.bar(i,self.do[i] , color="blue")

        return

    def plotWDI(self):
        plt.figure("WDI")
        plt.plot(range(len(self.X3)), [x[7] for x in self.X3])
        plt.plot(range(len(self.X3)), [x[8] for x in self.X3])
        plt.plot(range(len(self.X3)), [x[9] for x in self.X3])
        plt.plot(range(len(self.X3)), [x[10] for x in self.X3])

        return

    def plotdeepsSorted(self):
        deeps2 = OrderedDict(sorted(self.d2.items(), key=operator.itemgetter(1)))
        deeps = OrderedDict(sorted(self.d5.items(), key=operator.itemgetter(1)))
        pnj1 = self.w5[-1]
        plt.figure(1)

        x = []
        y = []
        y1 = []
        for r in deeps:
            x.append(r[0])
            y.append(r[1])
            y1.append(self.wn5[r[0]] * 100)

        plt.subplot(211)
        plt.bar(range(1, 51), y, color="blue", align='center', width=0.25)
        plt.bar(range(1, 51), y1, color="red", align='center', width=0.25)
        plt.xticks(np.arange(len(x)) + 0.5, x, rotation=90)

        x = []
        y = []
        y1 = []
        for r in deeps2:
            x.append(r[0])
            y.append(r[1])
            y1.append(self.wn2[r[0]] * 100)

        plt.subplot(212)
        plt.bar(range(1, 51), y, color="blue", align='center', width=0.25)
        plt.bar(range(1, 51), y1, color="red", align='center', width=0.25)
        plt.xticks(np.arange(len(x)) + 0.5, x, rotation=90)

        return


    def plotWn5(self):
        plt.figure(1)
        sx = []
        sy = []
        sx1 = []
        sy1 = []
        for k, v in self.wn5.items():
            sx.append(k)
            sy.append(v * 200)
            sx1.append(k)
            sy1.append(self.d5[k])

        plt.subplot(211)
        plt.bar(range(1, 51), sy, color="blue", align='center', width=0.25)
        plt.bar(range(1, 51), sy1, color="red", align='center', width=0.25)
        plt.xticks(np.arange(len(sx)) + 0.5, sx, rotation=90)

        sx = []
        sy = []
        sx1 = []
        sy1 = []
        for k, v in self.wn2.items():
            sx.append(k)
            sy.append(v * 200)
            sx1.append(k)
            sy1.append(self.d2[k])

        plt.subplot(212)
        plt.bar(range(1, 51), sy, color="blue", align='center', width=0.25)
        plt.bar(range(1, 51), sy1, color="red", align='center', width=0.25)
        plt.xticks(np.arange(len(sx)) + 0.5, sx, rotation=90)

        return