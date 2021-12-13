import pandas as pd

#def export_results(results, fn_prefix):
#    for _key, data in results.items():
#        _filename = "{}-{}.dat".format(fn_prefix, _key)
#        _data = pd.DataFrame.from_dict(data)
#        _data.to_csv(_filename, sep="\t", index=False)

def export_results(results, filename):
    if not filename.endswith(".dat"):
        filename = "{}.dat".format(filename)
    _data = pd.DataFrame.from_dict(results)
    _data.to_csv(filename, sep="\t", index=False)

