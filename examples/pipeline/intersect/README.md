## Intersect Pipeline Example Usage Guide.

#### Example Tasks

This section introduces the Pipeline scripts for different types of tasks.

1. RSA Intersection:

    script: pipeline-intersect-rsa.py

2. RSA Intersection with Random Base Fraction set to 0.5:

    script: pipeline-intersect-rsa-fraction.py

3. RSA Intersection with Calculation Split:

    script: pipeline-intersect-rsa-split.py

4. RSA Multi-hosts Intersection:

    script: pipeline-intersect-multi-rsa.py

5. DH Intersection:
    script: pipeline-intersect-dh.py

6. DH Multi-host Intersection:  
    
    script: pipeline-intersect-dh-multi.py

7. ECDH Intersection:
    
   script: pipeline-intersect-ecdh.py

8. ECDH Intersection with Preprocessing:
    
   script: pipeline-intersect-ecdh-w-preprocess.py

9. ECDH Intersect of 200 Union Components as Input:
    
   script: pipeline-intersect-with-union.py

10. RSA Intersect with Cache:
    
    script: pipeline-intersect-rsa-cache.py
 
11. DH Intersect with Cache:
    
    script: pipeline-intersect-dh-cache.py   

12. ECDH Intersect with Cache:
    
    script: pipeline-intersect-ecdh-cache.py   
      
13. RSA Intersect with Cache Loader:
    
    script: pipeline-intersect-rsa-cache-loader.py
    
14. Estimated Intersect Cardinality with RSA:
    
    script: pipeline-intersect-rsa-cardinality.py

15. Exact Intersect Cardinality with ECDH:
    
    script: pipeline-intersect-ecdh-exact-cardinality.py

16. Exact Intersect Cardinality with DH:
    
    script: pipeline-intersect-dh-exact-cardinality.py

17. DH Intersection with Preprocessing:
    
    script: pipeline-intersect-dh-w-preprocess.py

18. RSA Intersection with Preprocessing:
    
    script: pipeline-intersect-rsa-w-preprocess.py

19. ECDH Intersect with Cache Loader:
    
    script: pipeline-intersect-ecdh-cache-loader.py   

20. Exact Multi-host Intersect Cardinality with ECDH:
    
    script: pipeline-intersect-ecdh-multi-exact-cardinality.py

21. Exact Multi-host Intersect Cardinality with DH:
    
    script: pipeline-intersect-dh-multi-exact-cardinality.py

22. Exact Multi-host Intersect with ECDH:
    
    script: pipeline-intersect-ecdh-multi.py


Users can run a pipeline job directly:

    python ${pipeline_script}
