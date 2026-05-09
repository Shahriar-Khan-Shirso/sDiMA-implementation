def calculate_plddt(predictions, index_list, device, pdb_path):
    results = {}
    if not os.path.exists(pdb_path):
        os.makedirs(pdb_path)

    for i, seq in zip(index_list, predictions):
        print(f">>> STARTING sequence {i} (Length: {len(seq)})...") 
        try:

            with torch.no_grad():
                pdb_string = esmfold.infer_pdb(seq)
            
            print(f"    [DEBUG] ESMFold finished for {i}")
            
            file_name = os.path.join(pdb_path, f"prediction_{i}.pdb")
            with open(file_name, "w") as f:
                f.write(pdb_string)
            
            struct = bsio.load_structure(file_name, extra_fields=['b_factor'])
            results[seq] = float(struct.b_factor.mean())
            print(f"    [OK] Score calculated: {results[seq]:.2f}")

        except Exception as e:
            print(f"    [ERROR] Folding sequence {i}: {e}")
            results[seq] = 0.0
            
    return results
