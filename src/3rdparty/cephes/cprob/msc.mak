# MSDOS Microsoft C makefile for probability integrals.
# Be sure to disable the XPD pad for long double constants
# and set the type of computer to IBMPC in mconf.h.
#
CC = cl

CFLAGS = /c
# For large memory model:
#CFLAGS=/c /AL

INCS = mconf.h

OBJS = bdtr.obj btdtr.obj chdtr.obj const.obj drand.obj fdtr.obj \
gamma.obj gdtr.obj igam.obj igami.obj incbet.obj incbi.obj \
mtherr.obj nbdtr.obj ndtr.obj ndtri.obj pdtr.obj polevl.obj \
stdtr.obj unity.obj

bdtr.obj: bdtr.c $(INCS)
	$(CC) $(CFLAGS) bdtr.c

btdtr.obj: btdtr.c $(INCS)
	$(CC) $(CFLAGS) btdtr.c

chdtr.obj: chdtr.c $(INCS)
	$(CC) $(CFLAGS) chdtr.c

const.obj: const.c $(INCS)
	$(CC) $(CFLAGS) const.c

drand.obj: drand.c $(INCS)
	$(CC) $(CFLAGS) drand.c

expx2.obj: expx2.c $(INCS)
	$(CC) $(CFLAGS) expx2.c

fdtr.obj: fdtr.c $(INCS)
	$(CC) $(CFLAGS) fdtr.c

gamma.obj: gamma.c $(INCS)
	$(CC) $(CFLAGS) gamma.c

gdtr.obj: gdtr.c $(INCS)
	$(CC) $(CFLAGS) gdtr.c

igam.obj: igam.c $(INCS)
	$(CC) $(CFLAGS) igam.c

igami.obj: igami.c $(INCS)
	$(CC) $(CFLAGS) igami.c

incbet.obj: incbet.c $(INCS)
	$(CC) $(CFLAGS) incbet.c

incbi.obj: incbi.c $(INCS)
	$(CC) $(CFLAGS) incbi.c

mtherr.obj: mtherr.c $(INCS)
	$(CC) $(CFLAGS) mtherr.c

nbdtr.obj: nbdtr.c $(INCS)
	$(CC) $(CFLAGS) nbdtr.c

ndtr.obj: ndtr.c $(INCS)
	$(CC) $(CFLAGS) ndtr.c

ndtri.obj: ndtri.c $(INCS)
	$(CC) $(CFLAGS) ndtri.c

pdtr.obj: pdtr.c $(INCS)
	$(CC) $(CFLAGS) pdtr.c

polevl.obj: polevl.c $(INCS)
	$(CC) $(CFLAGS) polevl.c

stdtr.obj: stdtr.c $(INCS)
	$(CC) $(CFLAGS) stdtr.c

unity.obj: unity.c $(INCS)
	$(CC) $(CFLAGS) unity.c

# Delete the library file before attempting to rebuild it.
prob.lib: $(OBJS)
	lib @msc.rsp
