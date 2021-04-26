import pandas as pd

def export_results(results, fn_prefix):
    for _key, data in results.items():
        _filename = "{}-{}.dat".format(fn_prefix, _key)
        _data = pd.DataFrame.from_dict(data)
        _data.to_csv(_filename, sep="\t", index=False)
