## Hetero Neural Network Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. Binary Train Task:

    script: pipeline-hetero-nn-train-binary.py

2. Multi-label Train Task:

    script: pipeline-hetero-nn-train-multi.py

3. Train Task With Early Stopping Strategy:

    script: pipeline-hetero-nn-train-with-early-stopping.py

4. Train Task With Selective BackPropagation Strategy:

    script: pipeline-hetero-nn-train-binary-selective-bp.py
    
5. Train Task With Interactive Layer DropOut Strategy:
    
    script: pipeline-hetero-nn-train-binary-drop-out.py
    
6. Train Task With Floating Point Precision Optimization:

    script: pipeline-hetero-nn-train-binary-floating_point_precision.py
    
7. Train Task With Warm Start:  
    
    script: pipeline-hetero-nn-train-with-warm_start.py  
    
8. Train Task With CheckPoint:  
 
    script: pipeline-hetero-nn-train-with-check-point.py  
    
Users can run a pipeline job directly:

    python ${pipeline_script}
