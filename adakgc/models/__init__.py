from models.spotasoc_constraint_decoder import SpotConstraintDecoder, SpotAsocConstraintDecoder

def get_constraint_decoder(tokenizer, record_schema, decoding_schema, task_name='event'):
    if decoding_schema == 'spotasoc':
        if len(record_schema.role_list) == 0:
            task_map = {
                'entity': SpotConstraintDecoder,
                'relation': SpotConstraintDecoder,
                'event': SpotConstraintDecoder,
                'record': SpotConstraintDecoder,
            }
        else:
            task_map = {
                'entity': SpotAsocConstraintDecoder,
                'relation': SpotAsocConstraintDecoder,
                'event': SpotAsocConstraintDecoder,
                'record': SpotAsocConstraintDecoder,
            }
    else:
        raise NotImplementedError(
            f'Type Schema {record_schema}, Decoding Schema {decoding_schema}, Task {task_name} do not map to constraint decoder.'
        )
    return task_map[task_name](tokenizer=tokenizer, record_schema=record_schema)
