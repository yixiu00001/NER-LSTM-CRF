import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG

def get_entity_medical(tag_seq, char_seq):
    DISEASE = get_DISEASE_entity(tag_seq, char_seq)
    SYMPTOM = get_SYMPTOM_entity(tag_seq, char_seq)
    BODY = get_BODY_entity(tag_seq, char_seq)
    return DISEASE, SYMPTOM, BODY

def get_DISEASE_entity(tag_seq, char_seq):
    length = len(char_seq)
    DIS = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-DISEASE':
            if 'dis' in locals().keys():
                DIS.append(dis)
                del dis
            dis = char
            if i+1 == length:
                DIS.append(dis)
        if tag == 'I-DISEASE':
            if 'dis' in locals().keys():
                dis += char  
            else:
                dis = char
            if i+1 == length:
                DIS.append(dis)
        if tag not in ['I-DISEASE', 'B-DISEASE']:
            if 'dis' in locals().keys():
                DIS.append(dis)
                del dis
            continue
    return DIS


def get_SYMPTOM_entity(tag_seq, char_seq):
    length = len(char_seq)
    SYM = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-SYMPTOM':
            if 'sym' in locals().keys():
                SYM.append(sym)
                del sym
            sym = char
            if i+1 == length:
                SYM.append(sym)
        if tag == 'I-SYMPTOM':
            if "sym" in locals().keys():
                sym += char
            else:
                sym = char
            if i+1 == length:
                SYM.append(sym)
        if tag not in ['I-SYMPTOM', 'B-SYMPTOM']:
            if 'sym' in locals().keys():
                SYM.append(sym)
                del sym
            continue
    return SYM


def get_BODY_entity(tag_seq, char_seq):
    length = len(char_seq)
    BODY = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-BODY':
            if 'body' in locals().keys():
                BODY.append(body)
                del body
            body = char
            if i+1 == length:
                BODY.append(body)
        if tag == 'I-BODY':
            if 'body' in locals().keys():
                body += char
            else:
                body = char
            if i+1 == length:
                BODY.append(body)
        if tag not in ['I-BODY', 'B-BODY']:
            if 'body' in locals().keys():
                BODY.append(body)
                del body
            continue
    return BODY

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
