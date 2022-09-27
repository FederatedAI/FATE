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

10. ECDH Intersection:
    
    script: pipeline-intersect-ecdh.py

11. ECDH Intersection with Preprocessing:
    
    script: pipeline-intersect-ecdh-w-preprocess.py

12. RAW Intersect of 200 Union Components as Input:
    
    script: pipeline-intersect-with-union.py

13. RSA Intersect with Cache:
    
    script: pipeline-intersect-rsa-cache.py
 
14. DH Intersect with Cache:
    
    script: pipeline-intersect-dh-cache.py   

15. ECDH Intersect with Cache:
    
    script: pipeline-intersect-ecdh-cache.py   
      
16. RSA Intersect with Cache Loader:
    
    script: pipeline-intersect-rsa-cache-loader.py
    
17. Estimated Intersect Cardinality with RSA:
    
    script: pipeline-intersect-rsa-cardinality.py

18. Exact Intersect Cardinality with ECDH:
    
    script: pipeline-intersect-ecdh-exact-cardinality.py

19. Exact Intersect Cardinality with DH:
    
    script: pipeline-intersect-dh-exact-cardinality.py

20. DH Intersection with Preprocessing:
    
    script: pipeline-intersect-dh-w-preprocess.py

21. RSA Intersection with Preprocessing:
    
    script: pipeline-intersect-rsa-w-preprocess.py

22. ECDH Intersect with Cache Loader:
    
    script: pipeline-intersect-ecdh-cache-loader.py   
     

Users can run a pipeline job directly:

    python ${pipeline_script}
