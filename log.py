import logging
def setup_logging(script_name):
    logger=logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)


    handler=logging.FileHandler(f'C:\\Users\\MURALI\\OneDrive\\Desktop\\Internship Projects\\Chunk prediction\\log_file\\{script_name}.log','w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger