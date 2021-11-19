## Intersect Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. RAW Intersection:

    script: pipeline-intersect-raw.py

2. RAW Intersection with SM3 Hashing:

    script: pipeline-intersect-raw-sm3.py

3. RSA Intersection:

    script: pipeline-intersect-rsa.py

4. RSA Intersection with Random Base Fraction set to 0.5:

    script: pipeline-intersect-rsa-fraction.py

5. RSA Intersection with Calculation Split:

    script: pipeline-intersect-rsa-split.py

6. RSA Multi-hosts Intersection:

    script: pipeline-intersect-multi-rsa.py

7. RSA Multi-hosts Intersection:

    script: pipeline-intersect-multi-raw.py

8. DH Intersection:

    script: pipeline-intersect-dh.py

9. DH Multi-host Intersection:  
    script: pipeline-intersect-dh-multi.py

10. RAW Intersect of 200 Union Components as Input:
    script: pipeline-intersect-with-union.py

11. RSA Intersect with Cache:
    script: pipeline-intersect-rsa-cache.py
 
12. DH Intersect with Cache:
    script: pipeline-intersect-dh-cache.py   
    
13. RSA Intersect with Cache Loader:
    script: pipeline-intersect-rsa-cache-loader.py
    
14. Intersect Cardinality:
    script": pipeline-intersect-cardinality.py
    
Users can run a pipeline job directly:

    python ${pipeline_script}
