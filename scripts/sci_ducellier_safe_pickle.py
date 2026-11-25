"""Safe(ish) loader for untrusted Ducellier pickles: restricted Unpickler with an
explicit whitelist of numpy/pandas/obspy classes needed to rebuild DataFrames and
obspy Traces. Also a scanner that lists the GLOBAL references without executing.
Run with python -I."""
import pickle, pickletools, sys, io

ALLOWED = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('pandas.core.frame', 'DataFrame'), ('pandas.core.series', 'Series'),
    ('pandas.core.internals.managers', 'BlockManager'),
    ('pandas.core.internals.managers', 'SingleBlockManager'),
    ('pandas.core.internals.blocks', 'new_block'),
    ('pandas._libs.internals', '_unpickle_block'),
    ('pandas.core.indexes.base', '_new_Index'),
    ('pandas.core.indexes.base', 'Index'),
    ('pandas.core.indexes.range', 'RangeIndex'),
    ('pandas.core.indexes.numeric', 'Int64Index'),
    ('pandas.core.indexes.numeric', 'Float64Index'),
    ('pandas._libs.tslibs.timestamps', '_unpickle_timestamp'),
    ('pandas.core.indexes.datetimes', '_new_DatetimeIndex'),
    ('pandas.core.indexes.datetimes', 'DatetimeIndex'),
    ('pandas.core.arrays.datetimes', 'DatetimeArray'),
    ('builtins', 'slice'), ('builtins', 'complex'), ('builtins', 'object'),
    ('copy_reg', '_reconstructor'), ('copyreg', '_reconstructor'),
    ('__builtin__', 'object'),
    ('obspy.core.trace', 'Trace'), ('obspy.core.trace', 'Stats'),
    ('obspy.core.utcdatetime', 'UTCDateTime'),
    ('obspy.core.util.attribdict', 'AttribDict'),
    ('obspy.core.stream', 'Stream'),
    ('collections', 'OrderedDict'), ('_codecs', 'encode'),
    ('obspy.core.inventory.response', 'CoefficientWithUncertainties'),
    ('obspy.core.inventory.response', 'CoefficientsTypeResponseStage'),
    ('obspy.core.inventory.response', 'FIRResponseStage'),
    ('obspy.core.inventory.response', 'FilterCoefficient'),
    ('obspy.core.inventory.response', 'InstrumentSensitivity'),
    ('obspy.core.inventory.response', 'PolesZerosResponseStage'),
    ('obspy.core.inventory.response', 'Response'),
    ('obspy.core.inventory.response', 'ResponseStage'),
    ('obspy.core.inventory.util', 'Azimuth'), ('obspy.core.inventory.util', 'Dip'),
    ('obspy.core.inventory.util', 'Frequency'),
    ('obspy.core.util.obspy_types', 'ComplexWithUncertainties'),
    ('obspy.core.util.obspy_types', 'FloatWithUncertaintiesAndUnit'),
    ('obspy.core.util.obspy_types', '_ComplexUncertainty'),
}

class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')

def load(path):
    with open(path, 'rb') as f:
        return Restricted(f, encoding='latin1').load()

def scan_globals(path):
    out = set(); strings = []
    with open(path, 'rb') as f:
        data = f.read()
    for op, arg, pos in pickletools.genops(data):
        if op.name in ('GLOBAL', 'INST'):
            out.add(tuple(arg.split(' ', 1)))
        elif op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'SHORT_BINSTRING', 'BINSTRING'):
            strings.append(arg)
        elif op.name == 'STACK_GLOBAL':
            out.add((strings[-2], strings[-1]))
    return out

if __name__ == '__main__':
    allg = {}
    for p in sys.argv[1:]:
        for g in scan_globals(p):
            allg.setdefault(g, 0); allg[g] += 1
    for g, n in sorted(allg.items()):
        print(n, g, 'OK' if g in ALLOWED else 'NOT-WHITELISTED')
