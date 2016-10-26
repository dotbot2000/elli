from __future__ import print_function
from fortranformat import FortranRecordReader as fread
from numpy import int32, float64, zeros, array
from BCTable import BCTable

class DSED_Isochrones:
    """Holds the contents of a Dartmouth isochrone file."""

    #reads in a file and sets a few basic quantities
    def __init__(self,filename,raw=False):
        self.filename=filename.strip()
        self.raw_format=raw
        try:
            if self.raw_format:
                self.read_raw_file()
            else:
                self.read_iso_file()
            self.columns=self.data[0].dtype.names
        except IOError:
            print("Failed to open isochrone file: ")
            print(self.filename)

    #this function reads in the raw isochrone format before magnitudes have been added
    def read_raw_file(self):
        #open file
        with open(self.filename,mode='r') as f:
            #define some line formats
            first_line=fread('(16X,I2)')
            third_line=fread('(1X,F7.4,F8.4,E11.4,E11.4,F7.2,F7.2)')
            #fifth_line=fread('(25x,A51)')
            age_eep_line=fread('(5X,F7.0,6X,I3)')
            column_line=fread('(1x,a3,1x,a4,6x,a4,7x,a7,4x,a7)')

            self.num_ages=first_line.read(f.readline())[0]
            self.num_cols = 5
            f.readline()
            self.mixl,self.Y,self.Z,Zeff,self.FeH,self.aFe=third_line.read(f.readline())

            ages=[]
            iso_set=[]
            for iage in range(self.num_ages):
                #read individual header and set up the isochrone container
                age,num_eeps=age_eep_line.read(f.readline())
                ages.append(age)
                names=column_line.read(f.readline())
                #do some polishing:
                names=[name.replace('/','_') for name in names]
                names=[name[0:name.find('.')] if name.find('.') > 0 else name for name in names]
                names=tuple(names)
                formats=tuple([int32]+[float64 for i in range(self.num_cols-1)])
                iso=zeros(num_eeps,{'names':names,'formats':formats})
                for eep in range(num_eeps):
                    x=f.readline().split()
                    y=[]
                    y.append(int(x[0]))
                    [y.append(z) for z in map(float64,x[1:])]
                    iso[eep]=tuple(y)
                if iage < self.num_ages-1:
                    f.readline()
                    f.readline()
                iso_set.append(iso)
            self.ages=array(ages)*1e-3 # convert Myr to Gyr
            self.data=array(iso_set)
            return True
            
    #this function reads in the standard isochrone format with mags
    def read_iso_file(self):
        #open file
        with open(self.filename,mode='r') as f:
            #define some line formats
            first_line=fread('(16X,I2,6X,I2)')
            third_line=fread('(1X,F7.4,F8.4,E11.4,E11.4,F7.2,F7.2)')
            fifth_line=fread('(25x,A51)')
            age_eep_line=fread('(5X,F6.3,6X,I3)')
            def column_line(mags): 
                return fread('(1x,a3,3x,a4,4x,a7,2x,a4,3x,a7,1x,{:d}a8)'.format(int(mags)))

            self.num_ages,num_mags=first_line.read(f.readline())
            self.num_cols = 5 + num_mags
            f.readline()
            f.readline()
            self.mixl,self.Y,self.Z,Zeff,self.FeH,self.aFe=third_line.read(f.readline())
            f.readline()
            self.system=fifth_line.read(f.readline())
            f.readline()

            ages=[]
            iso_set=[]
            for iage in range(self.num_ages):
                #read individual header and set up the isochrone container
                age,num_eeps=age_eep_line.read(f.readline())
                ages.append(age)
                names=column_line(num_mags).read(f.readline())
                #do some polishing:
                names=[name.replace('/','_') for name in names]
                names=[name[0:name.find('.')] if name.find('.') > 0 else name for name in names]
                names=tuple(names)
                formats=tuple([int32]+[float64 for i in range(self.num_cols-1)])
                iso=zeros(num_eeps,{'names':names,'formats':formats})
                for eep in range(num_eeps):
                    x=f.readline().split()
                    y=[]
                    y.append(int(x[0]))
                    [y.append(z) for z in map(float64,x[1:])]
                    iso[eep]=tuple(y)
                if iage < self.num_ages-1:
                    f.readline()
                    f.readline()
                iso_set.append(iso)
            self.ages=array(ages)
            self.data=array(iso_set)
            return True

    def find_age(self,age):
        my_age=float64(age)
        my_index=self.ages.searchsorted(my_age)
        if self.ages[my_index] == my_age:
            return self.data[my_index]
        else:
            print("Input age not in my list of ages.")
            return False

    def write_to_file(self,filename,filters=None):
        with open(filename,'w') as f:
            for iso in self.data:
                head1=''.join('{0:>10s}'.format(name) for name in iso.dtype.names[1:])
                head2=''.join('     {0:>5s}'.format(filter[:5].upper()) for filter in filters)
                f.write(head1+head2+'\n')
                for l in range(len(iso)):
                    if mags['f814wUVIS'][l]==mags['f814wUVIS'][l]:
                        data1=''.join('{0:10.5f}'.format(iso[name][l]) for name in iso.dtype.names[1:])
                        if filters:
                            data2=''.join('{0:10.5f}'.format(mags[filter][l]) for filter in filters)
                        else:
                            data2=''
                        f.write(data1+data2+'\n')
                f.write('\n')
