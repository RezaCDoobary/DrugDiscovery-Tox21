import drugdiscovery.dataintegrity as dataintegrity

if __name__ == "__main__":
    filename = 'data/tox21_10k_data_all.sdf'
    di = dataintegrity.DataIntegrity(filename)
    di.clean_columns(['FW'])
    target_columns = ['SR-HSE',
        'NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',
        'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
    di.change_types(target_columns, float)
    di.merge_duplicate_target_rows('SMILES',target_columns)
    di.save('data/data_dups_removed.sdf','data/data_dups_removed.csv')