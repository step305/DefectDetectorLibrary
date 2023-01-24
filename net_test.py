import multiprocessing as mp

ready = mp.Event()
result = mp.Queue()
test_proc = None


def test_body(net_type, model_path, out, event):
    net = net_type(model_path)
    res = net.verification()
    out.put(res)
    event.set()


def run_test(net_type, model_path):
    global test_proc
    test_proc = mp.Process(target=test_body, args=(net_type, model_path, result, ready))
    test_proc.start()


def is_ready():
    return ready.is_set()


def report():
    if ready.is_set():
        res = result.get()
        test_proc.join()
        return res
    else:
        return None
