from __future__ import print_function
from numpy import array,asscalar,concatenate,empty,float64
from numpy import genfromtxt,shape,squeeze,unique,where
from scipy.interpolate import griddata, pchip

class BCTable(object):
    """Holds a table of Teff, logg, and bolometric corrections."""

    def __init__(self,filename,skip=None):
        f=filename.strip()
        try:
            if skip:
                self.data=genfromtxt(f,names=True,dtype=None,
                                     skip_header=skip[0],skip_footer=skip[1])
                off=0
            else:
                self.data=genfromtxt(f,names=True,dtype=None,skip_header=1)
                off=0
        except IOError:
            print("Failed to open input file: ")
            print(f)

        grid=[]; mags=[]
        Teff=[]; logg=[]
        for d in self.data:
            Teff.append(d['Teff'])
            logg.append(d['logg'])
            grid.append([d['Teff'],d['logg']])
            dl=list(d)
            mags.append([dl[off:]])
        self.Teff=array(sorted(set(Teff)))
        self.logg=array(sorted(set(logg)))
        self.grid=array(grid)
        self.mags=array(mags)
        self.FeH=self.data[0]['FeH']
        self.names=self.data.dtype.names[off:]
        if skip:
            self.Av=self.data['Av'][0]
            self.Rv=self.data['Rv'][0]
        else:
            self.Av=0.0
            self.Rv=0.0
        self.SolBol = 4.75

    def get_mags(self,Teff,logg,logL):
        if shape(Teff)==shape(logg)==shape(logL):
            Teff=array(Teff,ndmin=1)
            logg=array(logg,ndmin=1)
            logL=array(logL,ndmin=1)
            formats=[float64 for name in self.names]
            BCs=squeeze(griddata(self.grid,self.mags,(Teff,logg),method='cubic'))

        mags=empty(shape(Teff),{'names':self.names,'formats':formats})
        for i in range(len(mags)):
            mags[i]=BCs[i]
            for name in self.names:
                if name not in ['Teff','logg','FeH','Av','Rv']:
                    mags[i][name]=self.SolBol-2.5*logL[i] - mags[i][name]
        return mags

    def table_check(self):
        count=0
        for i in range(1,len(self.grid)):
            Teff1=self.grid[i-1][0]
            Teff2=self.grid[i][0]
            if Teff1 == Teff2:
                logg1=self.grid[i-1][1]
                logg2=self.grid[i][1]
                dg=logg2-logg1
                if logg2 > logg1 and dg == 0.5:
                    pass
                else:
                    print(Teff1, Teff2)
                    print(logg1, logg2)
                    index=where((self.logg>logg1)&(self.logg<logg2))
                    gnew=self.logg[index]
                    tnew=Teff1
                    result=self.table_interp(tnew,gnew)[0][0]
                    print(result)
                    count+=1
        return count


class BC_Av_Table(object):
    """ Holds an expanded version of BCTable that specifies Av and Rv"""
    def __init__(self,filename):
        def count(filename):
            #first part gets the first line
            with open(filename) as f:
                f.readline()
                res=f.readline().split()
            #res[0] = '#'
            #res[1] = number of filters
            #res[2] = number of spectra
            #res[3] = number of Av
            #res[4] = number of Rv
            # => number of tables = res[3]*res[4]
            #second part gets the total number of lines
            lines=0
            for line in open(filename):
                lines += 1
            return int(res[2]), int(res[3])*int(res[4]), lines

        def start_of_table(tup,n):
            header=3
            tbl_head=2
            tbl_foot=0
            tbl_rows=tup[0]
            sot = header + (n-1)*(tbl_head+tbl_rows+tbl_foot) + 1
            return sot

        def end_of_table(tup,n):
            sot=start_of_table(tup,n)
            eot=sot+1+tup[0]
            return eot

        def skip_header_footer(tup,n):
            header=start_of_table(tup,n)
            finish=end_of_table(tup,n)
            footer=tup[2]-finish+2*(n-tup[1])
            return header, footer

        x=count(filename)

        self.lines_per_table=x[0]
        self.tables_per_file=x[1]
        self.lines_per_file =x[2]

        self.data=[]
        self.Av=[]
        self.Rv=[]

        for i in range(1,self.tables_per_file+1):
            header,footer=skip_header_footer(x,i)
            d=BCTable(filename,[header,footer])
            self.data.append(d)
            self.Av.append(d.Av)
            self.Rv.append(d.Rv)

        self.Av=array(self.Av)
        self.Rv=asscalar(unique(self.Rv))


    def get_mags(self,Teff,logg,logL):
        x=self.Av
        y=[]
        for d in self.data:
            y.append(d.get_mags(Teff,logg,logL))
        return concatenate(y)
