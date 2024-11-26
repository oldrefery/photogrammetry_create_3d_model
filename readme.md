# Images should be located 
in folder ```images``` in the root of the project

# Create docker image:
```
docker build -t 3d-scanner .
```

# Run from the beginning (with option --clean)
```
docker run --rm \
   --memory="16g" \
   --cpus="4" \
   -v "$(pwd)/images:/app/images:ro" \
   -v "$(pwd)/output:/app/output:rw" \
   -v "$(pwd)/database:/app/database:rw" \
   --tmpfs /tmp \
   3d-scanner python3 /app/src/create_3d_model.py --clean
```
