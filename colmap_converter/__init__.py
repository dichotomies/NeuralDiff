
def convert_file_format(x, format_trg="bmp"):
    if format_trg is None:
        return x
    fn, ext = os.path.splitext(x)
    return fn + os.path.extsep + format_trg
