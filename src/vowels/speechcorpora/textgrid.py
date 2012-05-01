#!/usr/bin/env python
# classes for Praat TextGrid data structures, and HTK .mlf fids
# Kyle Gorman <kgorman@ling.upenn.edu> (contributions from Morgan Sonderegger)
# modifications by Maarten Versteegh

#Copyright (c) 2011 Kyle Gorman, Max Bane, Morgan Sonderegger 
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in 
#the Software without restriction, including without limitation the rights to 
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
#of the Software, and to permit persons to whom the Software is furnished to do
#so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all 
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
#SOFTWARE.


from re import search
import types


class MLF(object):
    """ read in a HTK .mlf fid. iterating over it gives you a list of 
    TextGrids """


    def __init__(self, fid):
        self.items = []
        self.n = 0
        fid.readline() # get rid of header
        while 1: # loop over fid
            name = fid.readline()[1:-1]
            if name:
                grid = TextGrid()
                phon = IntervalTier('phones')
                word = IntervalTier('words')
                wmrk = ''
                wsrt = 0.
                wend = 0.
                while 1: # loop over the lines in each grid
                    line = fid.readline().rstrip().split()
                    if len(line) == 4: # word on this baby
                        pmin = line[0] / 10.e6
                        pmax = line[1] / 10.e6
                        phon.append_Interval(pmin, pmax, line[2])
                        if wmrk:
                            word.append_Interval(wsrt, wend, wmrk)
                        wmrk = line[3]
                        wsrt = pmin
                        wend = pmax
                    elif len(line) == 3: # just phone
                        pmin = line[0] / 10.e6
                        pmax = line[1] / 10.e6
                        phon.append_Interval(pmin, pmax, line[2])
                        wend = pmax 
                    else: # it's a period
                        word.append_Interval(wsrt, wend, wmrk)
                        self.items.append(grid)
                        break
                grid.append_Tier(phon)
                grid.append_Tier(word)
                self.n += 1
            else:
                break
        fid.close()


    def __iter__(self):
        return iter(self.items)


    def __len__(self):
        return self.n


    def __str__(self):
        return '<MLF instance with %d TextGrids>' % self.n


class TextGrid(object):
    """ represents Praat TextGrids as list of different types of tiers """


    def __init__(self, name=None, xmin=0, xmax=0): 
        self.tiers = []
        self.n = 0
        self.name = name # this is just for the MLF case
        self.xmin = xmin
        self.xmax = xmax
        
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.name != other.name:
            return False
        if self.n != other.n:
            return False
        if self.xmin != other.xmin:
            return False
        if self.xmax != other.xmax:
            return False
        if len(self.tiers) != len(other.tiers):
            return False
        for i in range(len(self.tiers)):
            match = (self.tiers[i] == other.tiers[i])
            if not match:
                return False
        return True
    
    def __str__(self):
        return '<TextGrid "%s" with %d tiers>' % (self.name, self.n)


    def __iter__(self):
        return iter(self.tiers)


    def __len__(self):
        return self.n


    def __getitem__(self, i):
        """ return the (i-1)th tier """
        if type(i) is types.IntType:
            return self.tiers[i]
        elif type(i) in types.StringTypes:
            return self.tiers[self._tier_map[i]]
        else:
            raise TypeError, 'argument must be of type int or str'
        
    def __setitem__(self, i, tier):
        if type(i) in types.IntType:
            self.tiers[i] = tier
        elif type(i) in types.StringTypes:
            self.tiers[self._tier_map[i]] = tier
        else:
            raise TypeError, 'argument must be of type int or str'

    def tier_names(self):
        return [t.name for t in self.tiers]

    def append_Tier(self, tier):
        self.tiers.append(tier)
        self.xmax = max(tier.xmax, self.xmax)
        self.n += 1


    def read(self, fid):
        fid.readline() # header crap
        fid.readline()
        fid.readline()
        self.xmin = float(fid.readline().rstrip().split()[2])
        self.xmax = float(fid.readline().rstrip().split()[2])
        fid.readline()
        m = int(fid.readline().rstrip().split()[2]) # will be self.n 
        fid.readline()
        for i in xrange(m): # loop over grids
            fid.readline()
            if fid.readline().rstrip().split()[2] == '"IntervalTier"': 
                iname  = fid.readline().rstrip().split('\"')[1]
                # iname = fid.readline().rstrip().split()[2][1:-1]                
                imin   = float(fid.readline().rstrip().split()[2])
                imax   = float(fid.readline().rstrip().split()[2])
                itime  = IntervalTier(iname, imin, imax) 
                n = int(fid.readline().rstrip().split()[3])
                for j in xrange(n):
                    fid.readline().rstrip().split() # header junk
                    jmin  = float(fid.readline().rstrip().split()[2])
                    jmax  = float(fid.readline().rstrip().split()[2])
                    match = search('(\S+) (=) (".*")', fid.readline().rstrip())
                    jmark = match.groups()[2][1:-1]
                    itime.append_Interval(jmin, jmax, jmark)
                self.append_Tier(itime) 
            else: # pointTier
                iname  = fid.readline().rstrip().split('\"')[1]                
                # iname = fid.readline().rstrip().split()[2][1:-1]
                imin  = float(fid.readline().rstrip().split()[2])
                imax  = float(fid.readline().rstrip().split()[2])
                itime = PointTier(iname, imin, imax) 
                n = int(fid.readline().rstrip().split()[3])
                for j in xrange(n):
                    fid.readline().rstrip() # header junk
                    jtime = float(fid.readline().rstrip().split()[2])
                    match = search('(\S+) (=) (".*")', fid.readline().rstrip())
                    jmark = match.groups()[2][1:-1]
                    itime.append_Point(jtime, jmark)
                self.append_Tier(itime)
        fid.close()
        self._tier_map = dict(zip([t.name for t in self.tiers], range(len(self.tiers))))


    def write(self, fid):
        """ write it into a text grid that Praat can read """
        fid.write('File type = "ooTextFile"\n')
        fid.write('Object class = "TextGrid"\n\n')
        fid.write('xmin = %.5f\n' % self.xmin)
        fid.write('xmax = %.5f\n' % self.xmax)
        fid.write('tiers? <exists>\n')
        fid.write('size = %d\n' % self.n)
        fid.write('item []:\n')
        for (tier, n) in zip(self.tiers, xrange(1, self.n + 1)):
            fid.write('\titem [%d]:\n' % n)
            if tier.__class__ == IntervalTier: 
                fid.write('\t\tclass = "IntervalTier"\n')
                fid.write('\t\tname = "%s"\n' % tier.name)
                fid.write('\t\txmin = %.5f\n' % tier.xmin)
                fid.write('\t\txmax = %.5f\n' % tier.xmax)
                fid.write('\t\tintervals: size = %d\n' % len(tier))
                for (interval, o) in zip(tier, xrange(1, len(tier) + 1)): 
                    fid.write('\t\t\tintervals [%d]:\n' % o)
                    fid.write('\t\t\t\txmin = %.5f\n' % interval.xmin)
                    fid.write('\t\t\t\txmax = %.5f\n' % interval.xmax)
                    fid.write('\t\t\t\ttext = "%s"\n' % interval.mark)
            else: # PointTier
                fid.write('\t\tclass = "TextTier"\n')
                fid.write('\t\tname = "%s"\n' % tier.name)
                fid.write('\t\txmin = %.5f\n' % tier.xmin)
                fid.write('\t\txmax = %.5f\n' % tier.xmax)
                fid.write('\t\tpoints: size = %d\n' % len(tier))
                for (point, o) in zip(tier, xrange(1, len(tier) + 1)):
                    fid.write('\t\t\tpoints [%d]:\n' % o)
                    fid.write('\t\t\t\ttime = %.5f\n' % point.time)
                    fid.write('\t\t\t\tmark = "%s"\n' % point.mark)
        fid.close()


class IntervalTier(object):
    """ represents IntervalTier as a list plus some features: min/max time, 
    size, and tier name """


    def __init__(self, name=None, xmin=0, xmax=0):
        self.n = 0
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.intervals = []
        
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.n != other.n:
            return False
        if self.name != other.name:
            return False
        if self.xmin != other.xmin:
            return False
        if self.xmax != other.xmax:
            return False
        if len(self.intervals) != len(other.intervals):
            return False
        for i in range(len(self.intervals)):
            match = self.intervals[i] == other.intervals[i]
            if not match:
                return False
        return True

    def __str__(self):
        return '<IntervalTier "%s" with %d points>' % (self.name, self.n)


    def __iter__(self):
        return iter(self.intervals)


    def __len__(self):
        return self.n


    def __getitem__(self, i):
        """ return the (i-1)th interval """
        return self.intervals[i]
    
    def __setitem__(self, i, item):
        self.intervals[i] = item


    def append_Interval(self, xmin, xmax, mark):
        self.intervals.append(Interval(xmin, xmax, mark))
        self.xmax = max(xmax, self.xmax)
        self.n += 1


    def read(self, fid):
        fid.readline() # header junk 
        fid.readline()
        fid.readline()
        self.xmin = float(fid.readline().rstrip().split()[2])
        self.xmax = float(fid.readline().rstrip().split()[2])
        n = int(fid.readline().rstrip().split()[3])
        for i in xrange(n):
            fid.readline().rstrip() # header
            imin  = float(fid.readline().rstrip().split()[2]) 
            imax  = float(fid.readline().rstrip().split()[2])
            match = search('(\S+) (=) (".*")', fid.readline().rstrip())
            imark = match.groups()[2][1:-1]
            self.append_Interval(imin, imax, imark)
        fid.close()


    def write(self, fid):
        fid.write('File type = "ooTextFile"\n')
        fid.write('Object class = "IntervalTier"\n\n')
        fid.write('xmin = %.5f\n' % self.xmin)
        fid.write('xmax = %.5f\n' % self.xmax)
        fid.write('intervals: size = %d\n' % self.n)
        for (interval, n) in zip(self.intervals, xrange(1, self.n + 1)):
            fid.write('intervals [%d]:\n' % n)
            fid.write('\txmin = %.5f\n' % interval.xmin)
            fid.write('\txmax = %.5f\n' % interval.xmax)
            fid.write('\ttext = "%s"\n' % interval.mark)
        fid.close()


class PointTier(object):
    """ represents PointTier (also called TextTier for some reason) as a list
    plus some features: min/max time, size, and tier name """


    def __init__(self, name=None, xmin=0, xmax=0):
        self.n = 0
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.points = []


    def __str__(self):
        return '<PointTier "%s" with %d points>' % (self.name, self.n)
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.n != other.n:
            return False
        if self.name != other.name:
            return False
        if self.xmin != other.xmin:
            return False
        if self.xmax != other.xmax:
            return False
        if len(self.points) != len(other.points):
            return False
        for i in range(len(self.points)):
            if self.points[i] != other.points:
                return False
        return True
        


    def __iter__(self):
        return iter(self.points)
    

    def __len__(self):
        return self.n

    
    def __getitem__(self, i):
        """ return the (i-1)th tier """
        return self.points[i]
    
    def __setitem__(self, i, item):
        self.points[i] = item


    def append_Point(self, time, mark):
        self.points.append(Point(time, mark))
        self.xmax = max(time, self.xmax)
        self.n += 1


    def read(self, fid):
        fid.readline() # header junk 
        fid.readline()
        fid.readline()
        self.xmin = float(fid.readline().rstrip().split()[2])
        self.xmax = float(fid.readline().rstrip().split()[2])
        n = int(fid.readline().rstrip().split()[3])
        for i in xrange(n):
            fid.readline().rstrip() # header
            itime = float(fid.readline().rstrip().split()[2])
            match = search('(\S+) (=) (".*")', fid.readline().rstrip())
            imark = match.groups()[2][1:-1]
            self.append_Point(itime, imark)
        fid.close()


    def write(self, fid):
        fid.write('File type = "ooTextFile"\n')
        fid.write('Object class = "TextTier"\n\n')
        fid.write('xmin = %.5f\n' % self.xmin)
        fid.write('xmax = %.5f\n' % self.xmax)
        fid.write('points: size = %d\n' % self.n)
        for (point, n) in zip(self.points, xrange(1, self.n + 1)):
            fid.write('points [%d]:\n' % n)
            fid.write('\ttime = %.5f\n' % point.time)
            fid.write('\tmark = "%s"\n' % point.mark)
        fid.close()


class Interval(object):
    """ represent an Interval """


    def __init__(self, xmin, xmax, mark):
        self.xmin = xmin
        self.xmax = xmax
        self.mark = mark
    

    def __str__(self):
        return '<Interval "%s" %.5f:%.5f>' % (self.mark, self.xmin,self.xmax)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.xmin != other.xmin:
            return False
        if self.xmax != other.xmax:
            return False
        if self.mark != other.mark:
            return False
        return True

    def bounds(self):
        return (self.xmin, self.xmax)


class Point(object):
    """ represent a Point """


    def __init__(self, time, mark):
        self.time = time
        self.mark = mark
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.time != other.time:
            return False
        if self.mark != other.mark:
            return False
        return True

    def __str__(self):
        return '<Point "%s" at %.5f>' % (self.mark, self.time)

## testing code ##
if __name__ == '__main__':

    from os import remove
    from string import lowercase

    # make a point tier
    N = 10
    my_point_tier = PointTier(' my "point" tier ', 0, N)
    for i in xrange(N):
        my_point_tier.append_Point(i + .5, lowercase[i] + '"')
    print my_point_tier
    my_point_tier.write(open('my_point_tier.PointTier', 'w'))
    my_point_tier = PointTier() # erase it in memory
    my_point_tier.read(open('my_point_tier.PointTier', 'r'))
    print my_point_tier
    for point in my_point_tier:
        print point
    print
    
    # make an interval tier
    N = 15
    my_interval_tier = IntervalTier(' my "interval" tier ', 0, N)
    for i in xrange(N):
        my_interval_tier.append_Interval(i, i + 1, ' "' + lowercase[i] + '" ')
    print my_interval_tier
    my_interval_tier.write(open('my_interval_tier.IntervalTier', 'w'))
    my_interval_tier = IntervalTier() # erase it in memory
    my_interval_tier.read(open('my_interval_tier.IntervalTier', 'r'))
    print my_interval_tier
    for point in my_interval_tier:
        print point
    print

    my_text_grid = TextGrid(' my "text" grid' ) 
    my_text_grid.append_Tier(my_point_tier)
    my_text_grid.append_Tier(my_interval_tier)
    print my_text_grid.tier_names()
    my_text_grid.write(open('my_text_grid.TextGrid', 'w'))
    my_text_grid = TextGrid() # erase it in memory
    my_text_grid.read(open('my_text_grid.TextGrid', 'r'))
    for tier in my_text_grid:
        print tier.name
        for pointerval in tier: # could be a point or interval one...
            print pointerval
        print
    print
    remove('my_point_tier.PointTier')
    remove('my_interval_tier.IntervalTier')
    remove('my_text_grid.TextGrid')
