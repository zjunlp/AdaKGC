from typing import Dict


def list_dictionary(d, n_tab=-1):
    if isinstance(d, list):
        for i in d:
            list_dictionary(i, n_tab)
    elif isinstance(d, dict):
        n_tab += 1
        for key, value in d.items():
            if key == '<end>':
                print("{}{}".format(" " * n_tab, key))
            else:
                print("{}{}".format(" " * n_tab, key))
                list_dictionary(value, n_tab)
    else:
        print("{}{}".format("\t" * n_tab, d))


def print_tree(tree):
    list_dictionary(tree)


def get_label_name_tree(label_name_list, tokenizer, end_symbol='<end>'):
    '''
    label_name_list:
        ["end position", "attack", "start position", "nominate", "charge indict", "transfer money", "transfer ownership", "release parole", "born", "sentence", "die", "demonstrate", "arrest jail", "transport", "elect", "start organization", "meet", "injure", "phone write", "merge organization", "declare bankruptcy", "trial hearing", "fine", "convict", "end organization", "sue", "divorce", "acquit", "appeal", "marry", "execute", "extradite", "pardon"]
        ["person", "place", "destination", "giver", "plaintiff", "instrument", "attacker", "agent", "origin", "victim", "seller", "vehicle", "buyer", "organization", "adjudicator", "defendant", "beneficiary", "artifact", "prosecutor", "recipient", "target", "entity"]
    '''

    sub_token_tree = dict()

    label_tree = dict()   # {"end position":[], "attack":[],...}
    for typename in label_name_list:
        after_tokenized = tokenizer.encode(typename, add_special_tokens=False)
        # label_tree[typename] = tokenizer.convert_ids_to_tokens(after_tokenized)
        label_tree[typename] = after_tokenized

    for _, sub_label_seq in label_tree.items():
        parent = sub_token_tree
        for value in sub_label_seq:
            if value not in parent:
                parent[value] = dict()
            parent = parent[value]

        parent[end_symbol] = None

    return sub_token_tree


def get_type_role_tree(type_role_dict, role_tree, tokenizer, end_symbol='<end>'):
    type_role_tree = dict()
    for typename, role_list in type_role_dict.items():
        typetoken = tokenizer.encode(typename, add_special_tokens=False)
        sub_token_tree = dict()
        for role in role_list:
            role_token = tokenizer.encode(role, add_special_tokens=False)
            sub_token_tree.update({role_token[0]: role_tree[role_token[0]]})
        type_role_tree[tuple(typetoken)] = sub_token_tree
    return type_role_tree



class PrefixTree:
    def __init__(self, label_name_list, tokenizer, end_symbol='<end>'):
        self.label_name_list = label_name_list
        self._tokenizer = tokenizer
        self.label_name_tree = get_label_name_tree(label_name_list, tokenizer, end_symbol)
        self._end_symbol = end_symbol

    def is_end_of_tree(self, tree: Dict):
        return len(tree) == 1 and self._end_symbol in tree
