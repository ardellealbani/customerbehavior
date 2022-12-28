"""
This is a boilerplate pipeline 'data_prepocessing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preparation, preparation_oot, declare_target_class, declare_target_class_oot, encoding_target_class, encoding_target_class_oot, feature_selection_CN, feature_selection_CR, feature_selection_TN, feature_selection_TR, split_data, upsample_CN, upsample_CR, upsample_TN, upsample_TR, split_xy_CN, split_xy_CR, split_xy_TN, split_xy_TR, modeling_CN, modeling_CR, modeling_TN, modeling_TR, evaluating_CN, evaluating_CR, evaluating_TN, evaluating_TR,oot_CN, oot_CR, oot_TN, oot_TR, combine_oot, combine

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preparation,
            inputs=["dataset_raw","params:columns"],
            outputs="dataset",
            name="preparation_node"
        ),
        node(
            func=preparation_oot,
            inputs=["dataset_oot","params:columns"],
            outputs="oot_dataset",
            name="preparation_oot_node"
        ),        
        node(
            func=declare_target_class,
            inputs="dataset",
            outputs="dataset_target",
            name="target_class_node"
        ),
        node(
            func=declare_target_class_oot,
            inputs="oot_dataset",
            outputs="oot_dataset_target",
            name="target_class_oot_node"
        ),        
        node(
            func=encoding_target_class,
            inputs="dataset_target",
            outputs="dataset_encoding",
            name="encoding_node"
        ),
        node(
            func=encoding_target_class_oot,
            inputs="oot_dataset_target",
            outputs="oot_dataset_encoding",
            name="encoding_oot_node"
        ),
        node(
            func=split_data,
            inputs="dataset_encoding",
            outputs=["training","testing"],
            name="split_data_node"
        ),        
        node(
            func=feature_selection_CN,
            inputs="training",
            outputs="df_model_CN",
            name="feature_selection_CN_node"
        ),
        node(
            func=feature_selection_CR,
            inputs="training",
            outputs="df_model_CR",
            name="feature_selection_CR_node"
        ),
        node(
            func=feature_selection_TN,
            inputs="training",
            outputs="df_model_TN",
            name="feature_selection_TN_node"
        ),
        node(
            func=feature_selection_TR,
            inputs="training",
            outputs="df_model_TR",
            name="feature_selection_TR_node"
        ),                        
        node(
            func=upsample_CN,
            inputs="df_model_CN",
            outputs="training_CN_boostrap",
            name="upsample_CN_node"
        ),
        node(
            func=upsample_CR,
            inputs="df_model_CR",
            outputs="training_CR_boostrap",
            name="upsample_CR_node"
        ), 
        node(
            func=upsample_TN,
            inputs="df_model_TN",
            outputs="training_TN_boostrap",
            name="upsample_TN_node"
        ),
        node(
            func=upsample_TR,
            inputs="df_model_TR",
            outputs="training_TR_boostrap",
            name="upsample_TR_node"
        ),                      
        node(
            func=split_xy_CN,
            inputs=["training",'testing'],
            outputs=["CN_X_train", "CN_y_train", "CN_X_test", "CN_y_test"],
            name="split_xy_CN_node"
        ),
        node(
            func=split_xy_CR,
            inputs=["training",'testing'],
            outputs=["CR_X_train", "CR_y_train", "CR_X_test", "CR_y_test"],
            name="split_xy_CR_node"
        ),  
        node(
            func=split_xy_TN,
            inputs=["training",'testing'],
            outputs=["TN_X_train", "TN_y_train", "TN_X_test", "TN_y_test"],
            name="split_xy_TN_node"
        ), 
        node(
            func=split_xy_TR,
            inputs=["training",'testing'],
            outputs=["TR_X_train", "TR_y_train", "TR_X_test", "TR_y_test"],
            name="split_xy_TR_node"
        ),               
        node(
            func=modeling_CN,
            inputs=["CN_X_train", "CN_y_train", "CN_X_test", "CN_y_test"],
            outputs=["CN_training_results_cat", "CN_testing_results_cat","model_CN"],
            name="modeling_CN_node"
        ),
        node(
            func=modeling_CR,
            inputs=["CR_X_train", "CR_y_train", "CR_X_test", "CR_y_test"],
            outputs=["CR_training_results_cat", "CR_testing_results_cat","model_CR"],
            name="modeling_CR_node"
        ),
        node(
            func=modeling_TN,
            inputs=["TN_X_train", "TN_y_train", "TN_X_test", "TN_y_test"],
            outputs=["TN_training_results_cat", "TN_testing_results_cat","model_TN"],
            name="modeling_TN_node"
        ),
        node(
            func=modeling_TR,
            inputs=["TR_X_train", "TR_y_train", "TR_X_test", "TR_y_test"],
            outputs=["TR_training_results_cat", "TR_testing_results_cat","model_TR"],
            name="modeling_TR_node"
        ),
        node(
            func=evaluating_CN,
            inputs=["CN_training_results_cat", "CN_testing_results_cat"],
            outputs="metrics_CN",
            name="evaluation_CN_node"
        ),
        node(
            func=evaluating_CR,
            inputs=["CR_training_results_cat", "CR_testing_results_cat"],
            outputs="metrics_CR",
            name="evaluation_CR_node"
        ),
        node(
            func=evaluating_TN,
            inputs=["TN_training_results_cat", "TN_testing_results_cat"],
            outputs="metrics_TN",
            name="evaluation_TN_node"
        ),
        node(
            func=evaluating_TR,
            inputs=["TR_training_results_cat", "TR_testing_results_cat"],
            outputs="metrics_TR",
            name="evaluation_TR_node"
        ),
        node(
            func=oot_CN,
            inputs=["oot_dataset_encoding", "model_CN"],
            outputs=["CN_prediction", "CN_probability"],
            name="oot_CN_node"
        ), 
        node(
            func=oot_CR,
            inputs=["oot_dataset_encoding", "model_CR"],
            outputs=["CR_prediction", "CR_probability"],
            name="oot_CR_node"
        ), 
        node(
            func=oot_TN,
            inputs=["oot_dataset_encoding", "model_TN"],
            outputs=["TN_prediction", "TN_probability"],
            name="oot_TN_node"
        ),
        node(
            func=oot_TR,
            inputs=["oot_dataset_encoding", "model_TR"],
            outputs=["TR_prediction", "TR_probability"],
            name="oot_TR_node"
        ),
        node(
            func=combine_oot,
            inputs=["oot_dataset_encoding", "CN_prediction", "CR_prediction", "TN_prediction","TR_prediction", "CN_probability", "CR_probability", "TN_probability", "TR_probability"],
            outputs="oot",
            name="combine_oot_node"
        ),
        node(
            func=combine,
            inputs=["training", "testing", "model_CN", "model_CR","model_TN", "model_TR"],
            outputs=["training_4k", "testing_1k"],
            name="combine_node"
        )                                  
    ])
