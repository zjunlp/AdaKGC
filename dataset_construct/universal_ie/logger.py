import logging
import os

def init_logger(task_name):
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO) 

    logfile = './log/' + task_name + '_logger.txt'
    os.makedirs('./log/', exist_ok=True)   
   
    fh = logging.FileHandler(logfile, mode = 'a', encoding = 'utf-8')
    fh.setLevel(logging.INFO)    

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  

    formatter = logging.Formatter("%(asctime)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger