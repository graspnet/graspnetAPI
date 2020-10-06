def get_config():
    '''
     - return the config dict
    '''
    config = dict()
    force_closure = dict()
    force_closure['quality_method'] = 'force_closure'
    force_closure['num_cone_faces'] = 8
    force_closure['soft_fingers'] = 1
    force_closure['quality_type'] = 'quasi_static'
    force_closure['all_contacts_required']= 1
    force_closure['check_approach'] = False
    force_closure['torque_scaling'] = 0.01
    force_closure['wrench_norm_thresh'] = 0.001
    force_closure['wrench_regularizer'] = 0.0000000001
    config['metrics'] = dict()
    config['metrics']['force_closure'] = force_closure
    return config