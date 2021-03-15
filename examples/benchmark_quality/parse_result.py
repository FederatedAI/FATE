def parse_summary_result(rs_dict):
    for model_key in rs_dict:
        rs_content = rs_dict[model_key]
        if 'validate' in rs_content:
            return rs_content['validate']
        else:
            return rs_content['train']