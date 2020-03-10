import pandas
import numpy as np
import scipy
from scipy import stats
from numpy import median
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


data_prefix = "brca_metabric/"
model_path = "autoEncoderData/"


cna_data_path = data_prefix + "data_CNA.txt"
rna_data_path = data_prefix + "data_mRNA_median_Zscores.txt"
gene_data_path = data_prefix + "data_expression_median.txt"
patient_data_path = data_prefix + "data_clinical_patient.txt"
model_name = model_path + "my_model"


# Compute entropy for CNA variables
def entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    counts = counts / sum(counts)
    return scipy.stats.entropy(counts)


# Compute Median Absolute Deviation for RNA variables
def MAD(x):
    return median(abs(x - median(x)))



# Group CNA variables
def normalize_cna(x):
    if x == -1 or x == -2:
        x = -1
    elif x == 1 or x == 2:
        x = 1
    else:
        x = 0
    return x


# Define the set of PAM50 genes
PAM50_genes = ['FOXC1', 'MIA', 'KNTC2', 'CEP55', 'ANLN',
               'MELK', 'GPR160', 'TMEM45B',
               'ESR1', 'FOXA1', 'ERBB2', 'GRB7',
               'FGFR4', 'BLVRA', 'BAG1', 'CDC20',
               'CCNE1', 'ACTR3B', 'MYC', 'SFRP1',
               'KRT17', 'KRT5', 'MLPH', 'CCNB1', 'CDC6',
               'TYMS', 'UBE2T', 'RRM2', 'MMP11',
               'CXXC5', 'ORC6L', 'MDM2', 'KIF2C', 'PGR',
               'MKI67', 'BCL2', 'EGFR', 'PHGDH',
               'CDH3', 'NAT1', 'SLC39A6',
               'MAPT', 'UBE2C', 'PTTG1', 'EXO1', 'CENPF',
               'CDCA1', 'MYBL2', 'BIRC5']

data = []


def train_graph():

    # Load patient data from file
    patient_data = pandas.read_csv(patient_data_path, sep="\t", skiprows=[0, 1, 2, 3])
    patient_data_for_training = pandas.read_csv(patient_data_path, index_col='PATIENT_ID', sep="\t", skiprows=[0, 1, 2, 3])\
        .fillna('NA')\
        .replace('claudin-low', 'claudinLow')\
        .replace('Ductal/NST', 'DuctalNST')\
        .replace('Tubular/ cribriform', 'TubularCribriform') \
        .replace('BREAST CONSERVING', 'BREASTCONSERVING') \
        .replace('ER-/HER2-', 'ERHER2Neg') \
        .replace('ER+/HER2- High Prolif', 'ERHER2NegHigh') \
        .replace('ER+/HER2- Low Prolif', 'ERHER2NegLow') \
        .replace('HER2+', 'HER2Pos')\
        .replace('4ER+', 21) \
        .replace('4ER-', 22)
    patient_data_for_training.HORMONE_THERAPY = patient_data_for_training.HORMONE_THERAPY.map(dict(YES=1, NO=0, NA=0))
    patient_data_for_training.CELLULARITY = patient_data_for_training.CELLULARITY.map(dict(High=3, Moderate=2, Low=1, NA=2))
    patient_data_for_training.CHEMOTHERAPY = patient_data_for_training.CHEMOTHERAPY.map(dict(YES=1, NO=0, NA=0))
    patient_data_for_training.ER_IHC = patient_data_for_training.ER_IHC.map(dict(Positve=2, Negative=1, NA=1.5))
    patient_data_for_training.HER2_SNP6 = patient_data_for_training.HER2_SNP6.map(dict(NEUTRAL=2, GAIN=3, LOSS=1, UNDEF=2, NA=2))
    patient_data_for_training.INFERRED_MENOPAUSAL_STATE = patient_data_for_training.INFERRED_MENOPAUSAL_STATE.map(dict(Post=1, Pre=2, NA=1.5))
    patient_data_for_training.OS_STATUS = patient_data_for_training.OS_STATUS.map(dict(LIVING=1, DECEASED=2, NA=10))
    patient_data_for_training.CLAUDIN_SUBTYPE = patient_data_for_training.CLAUDIN_SUBTYPE.map(dict(Basal=1, claudinLow=2, Her2=3, LumA=4, LumB=5, NC=6, Normal=0, NA=1.5))
    patient_data_for_training.LATERALITY = patient_data_for_training.LATERALITY.map(dict(Right=1, Left=2, NA=1.5))
    patient_data_for_training.RADIO_THERAPY = patient_data_for_training.RADIO_THERAPY.map(dict(YES=1, NO=2, NA=1.5))
    patient_data_for_training.HISTOLOGICAL_SUBTYPE = patient_data_for_training.HISTOLOGICAL_SUBTYPE.map(
        dict(DuctalNST=1, Lobular=2, Medullary=3, Metaplastic=4, Mixed=5, Mucinous=6, NA=7, Other=8, TubularCribriform=9)
    )
    patient_data_for_training.BREAST_SURGERY = patient_data_for_training.BREAST_SURGERY.map(dict(BREASTCONSERVING=1, MASTECTOMY=2, NA=1.5))
    patient_data_for_training.THREEGENE = patient_data_for_training.THREEGENE.map(dict(ERHER2Neg=1, ERHER2NegHigh=2, ERHER2NegLow=3, HER2Pos=4, NA=2.5))

    patient_data_for_training = patient_data_for_training.replace('NA', 0)

    data_for_labels = pandas.DataFrame(index=patient_data_for_training.index, columns=["vital_status", "time_since_detection"])
    data_for_labels.vital_status = patient_data_for_training.OS_STATUS
    data_for_labels.time_since_detection = patient_data_for_training.OS_MONTHS
    data_for_labels = data_for_labels.transpose()

    label_sample = data_for_labels['MB-0000'].values.transpose()

    patient_data_for_training = patient_data_for_training.drop('VITAL_STATUS', axis=1)
    patient_data_for_training = patient_data_for_training.drop('OS_STATUS', axis=1)
    patient_data_transpose = patient_data_for_training.transpose()

    intclust_data = patient_data[['PATIENT_ID', 'INTCLUST']].dropna()

    # Load CNA data from file
    cna_data = pandas.read_csv(cna_data_path, sep="\t").dropna()
    cna_data = cna_data.drop(['Entrez_Gene_Id'], axis=1)

    # Load RNA data from file
    rna_data = pandas.read_csv(rna_data_path, sep="\t").dropna()
    rna_data = rna_data.drop(['Entrez_Gene_Id'], axis=1)

    # Load gene data from file
    gene_data = pandas.read_csv(gene_data_path, sep="\t").dropna()
    gene_data = gene_data.drop(['Entrez_Gene_Id'], axis=1)

    # Extract common genes
    common_genes = set(cna_data['Hugo_Symbol']) & set(rna_data['Hugo_Symbol']) & set(gene_data['Hugo_Symbol'])
    common_with_PAM50 = common_genes & set(PAM50_genes)
    common_genes = pandas.Series(list(common_genes)).dropna()
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(common_genes)]
    rna_data = rna_data.loc[rna_data['Hugo_Symbol'].isin(common_genes)]
    gene_data = gene_data.loc[gene_data['Hugo_Symbol'].isin(common_genes)]

    # Extract common patients
    common_cols = cna_data.columns.intersection(rna_data.columns).intersection(gene_data.columns)
    cna_data = cna_data[common_cols]
    rna_data = rna_data[common_cols]
    gene_data = gene_data[common_cols]

    # Sort by gene
    cna_data = cna_data.sort_values(by='Hugo_Symbol')
    rna_data = rna_data.sort_values(by='Hugo_Symbol')
    gene_data = gene_data.sort_values(by='Hugo_Symbol')

    # Extract most high-varied genes
    np_gene_data = rna_data.iloc[:, 1:].values
    top_MAD_cna = np.argsort(np.apply_along_axis(func1d=MAD, axis=1, arr=np_gene_data))[-1200:]

    # For random selection:
    # np.random.shuffle(top_MAD_cna)
    # top_MAD_cna = top_MAD_cna[:1200]

    # Obtain list of genes to extract
    selected_genes = cna_data.iloc[top_MAD_cna, 0]
    selected_genes = list(set(selected_genes) | common_with_PAM50)
    selected_genes = pandas.Series(list(selected_genes)).dropna()
    rna_data = rna_data.loc[rna_data['Hugo_Symbol'].isin(selected_genes)]
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(selected_genes)]
    gene_data = gene_data.loc[gene_data['Hugo_Symbol'].isin(selected_genes)]

    np_gene_data = cna_data.iloc[:, 1:].values
    top_MAD_cna = np.argsort(np.apply_along_axis(func1d=entropy, axis=1, arr=np_gene_data))[-300:]

    # For random selection:
    # np.random.shuffle(top_MAD_cna)
    # top_MAD_cna = top_MAD_cna[:300]

    selected_genes = cna_data.iloc[top_MAD_cna, 0]
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(selected_genes)]

    # Convert CNA to one-hot encoding
    cna_data = cna_data.iloc[:, 1:]
    cna_data = cna_data.applymap(normalize_cna)
    cna_data = cna_data.transpose()
    cna_data = pandas.get_dummies(cna_data, columns=cna_data.columns)
    cna_data = cna_data.transpose()

    # Remove gene column from RNA
    rna_data = rna_data.iloc[:, 1:]


    # Get number of features
    n_cna_features = cna_data.shape[0]
    n_rna_features = rna_data.shape[0]
    n_gene_features = gene_data.shape[0]
    print("CNA features: ", n_cna_features)
    print("RNA features: ", n_rna_features)
    print("Gene features: ", n_gene_features)

    np_type_data = []
    np_rna_data = []
    np_cna_data = []
    np_gene_data = []
    np_clinical_data = []
    np_labels = []

    for index, row in intclust_data.iterrows():

        patient_id = row['PATIENT_ID']
        cluster_id = row['INTCLUST']

        # Merge cluster 4
        if cluster_id == '4ER+' or cluster_id == '4ER-':
            cluster_id = 4

        # Exclude clusters 2 and 6
        if cluster_id == '2' or cluster_id == '6':
            continue

        cluster_id = int(cluster_id) - 1

        if patient_id in rna_data:

            # Check if number of elements per cluster is exceeded
            unique, counts = np.unique(np_type_data, return_counts=True)
            count_dict = dict(zip(unique, counts))
            if cluster_id in count_dict and count_dict[cluster_id] >= 200:
                continue

            rna_sample = rna_data[patient_id].values.transpose()
            gene_sample = gene_data[patient_id].values.transpose()
            cna_sample = cna_data[patient_id].values.transpose()
            clinical_sample = patient_data_transpose[patient_id].values.transpose()
            label_sample = data_for_labels[patient_id].values.transpose()
            if label_sample[0] == 2 and label_sample[1] < 60:
                np_labels.append(1)
            else:
                np_labels.append(0)

            np_rna_data.append(rna_sample)
            np_gene_data.append(gene_sample)
            np_cna_data.append(cna_sample)
            np_type_data.append(cluster_id)
            np_clinical_data.append(clinical_sample)

    np_rna_data = np.array(np_rna_data)
    np_gene_data = np.array(np_gene_data)
    np_cna_data = np.array(np_cna_data)
    np_type_data = np.array(np_type_data)
    np_clinical_data = np.array(np_clinical_data)
    np_labels = np.array(np_labels)

    # Normalize RNA data
    np_rna_data = 2 * (np_rna_data - np.min(np_rna_data)) / (np.max(np_rna_data) - np.min(np_rna_data)) - 1

    # Normalize gene data
    np_gene_data = 2 * (np_gene_data - np.min(np_gene_data)) / (np.max(np_gene_data) - np.min(np_gene_data)) - 1

    # Print cluster counts
    unique, counts = np.unique(np_type_data, return_counts=True)
    print(counts)

    # Split into training and test data
    n_samples = np_rna_data.shape[0]
    n_train_samples = int(n_samples * 0.8)
    sample_indices = np.arange(n_samples)
    np.random.shuffle(sample_indices)
    train_indices = sample_indices[:n_train_samples]
    test_indices = sample_indices[n_train_samples:]


    X_train_rna = np_rna_data[train_indices, :].copy()
    X_train_gene = np_gene_data[train_indices, :].copy()
    X_train_cna = np_cna_data[train_indices, :].copy()
    X_train_clinical_data = np_clinical_data[train_indices, :].copy()
    y_train = np_type_data[train_indices].copy()
    label_train = np_labels[train_indices].copy()

    X_test_rna = np_rna_data[test_indices, :].copy()
    X_test_gene = np_gene_data[test_indices, :].copy()
    X_test_cna = np_cna_data[test_indices, :].copy()
    X_test_clinical_data = np_clinical_data[test_indices, :].copy()
    y_test = np_type_data[test_indices].copy()
    label_test = np_labels[test_indices].copy()

    # For setting random RNA genes to zero:
    for i in range(X_test_rna.shape[0]):
        zero_indices = np.arange(1200)
        np.random.shuffle(zero_indices)
        zero_indices = zero_indices[:120]
        X_test_rna[i:i+1, zero_indices] = 0

    # For setting random RNA genes to zero:
    for i in range(X_test_gene.shape[0]):
        zero_indices = np.arange(1200)
        np.random.shuffle(zero_indices)
        zero_indices = zero_indices[:120]
        X_test_rna[i:i + 1, zero_indices] = 0


# ----------------------------------------Multi-Modal AutoEncoder---------------------------------------------------

    def run_multi_encoder(n_multi_epochs, verb):

        # Define layers
        rna_hidden = 800
        input_rna = Input(shape=(n_rna_features,))
        hidden_rna_layer_1 = Dense(rna_hidden, activation='sigmoid')

        gene_hidden = 800
        input_gene = Input(shape=(n_gene_features,))
        hidden_gene_layer_1 = Dense(gene_hidden, activation='sigmoid')

        cna_hidden = 800
        input_cna = Input(shape=(n_cna_features,))
        hidden_cna_layer_1 = Dense(cna_hidden, activation='sigmoid')

        enc_features = 1600
        combined_layer = Dense(enc_features, activation='sigmoid')

        hidden_rna_layer_2 = Dense(rna_hidden, activation='sigmoid')
        output_rna_layer = Dense(n_rna_features, activation='sigmoid')

        hidden_gene_layer_2 = Dense(gene_hidden, activation='sigmoid')
        output_gene_layer = Dense(n_gene_features, activation='sigmoid')

        hidden_cna_layer_2 = Dense(cna_hidden, activation='sigmoid')
        output_cna_layer = Dense(n_cna_features, activation='sigmoid')


        # Train first set of layers
        hidden_rna = hidden_rna_layer_1(input_rna)
        output_rna = output_rna_layer(hidden_rna)
        autoencoder = Model(input_rna, output_rna)
        autoencoder.compile(loss='mse', optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit(X_train_rna, X_train_rna,
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        hidden_rna_layer_1.trainable = False
        rna_hidden_encoder = Model(input_rna, hidden_rna)
        intermediate_rna = rna_hidden_encoder.predict(X_train_rna)

        hidden_gene = hidden_gene_layer_1(input_gene)
        output_gene = output_gene_layer(hidden_gene)
        autoencoder = Model(input_gene, output_gene)
        autoencoder.compile(loss='mse', optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit(X_train_gene, X_train_gene,
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        hidden_gene_layer_1.trainable = False
        gene_hidden_encoder = Model(input_gene, hidden_gene)
        intermediate_gene = gene_hidden_encoder.predict(X_train_gene)

        hidden_cna = hidden_cna_layer_1(input_cna)
        output_cna = output_cna_layer(hidden_cna)
        autoencoder = Model(input_cna, output_cna)
        autoencoder.compile(loss='mse', optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit(X_train_cna, X_train_cna,
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        hidden_cna_layer_1.trainable = False
        cna_hidden_encoder = Model(input_cna, hidden_cna)
        intermediate_cna = cna_hidden_encoder.predict(X_train_cna)


        # Train combined layer
        hidden_rna = hidden_rna_layer_1(input_rna)
        hidden_gene = hidden_gene_layer_1(input_gene)
        hidden_cna = hidden_cna_layer_1(input_cna)
        concat = Concatenate()([hidden_rna, hidden_cna, hidden_gene])
        combined = combined_layer(concat)
        output_rna = hidden_rna_layer_2(combined)
        output_gene = hidden_gene_layer_2(combined)
        output_cna = hidden_cna_layer_2(combined)
        autoencoder = Model([input_rna, input_cna, input_gene], [output_rna, output_cna, output_gene])
        autoencoder.compile(loss = ['mse', 'mse', 'mse'] , optimizer=optimizers.SGD(lr=0.01))
        autoencoder.fit([X_train_rna, X_train_cna, X_train_gene], [intermediate_rna, intermediate_cna, intermediate_gene],
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)
        combined_layer.trainable = False



        # Train full model
        hidden_rna = hidden_rna_layer_1(input_rna)
        hidden_gene = hidden_gene_layer_1(input_gene)
        hidden_cna = hidden_cna_layer_1(input_cna)
        concat = Concatenate()([hidden_rna, hidden_cna, hidden_gene])
        combined = combined_layer(concat)
        hidden_rna_2 = hidden_rna_layer_2(combined)
        hidden_gene_2 = hidden_gene_layer_2(combined)
        hidden_cna_2 = hidden_cna_layer_2(combined)
        output_rna = output_rna_layer(hidden_rna_2)
        output_gene = output_gene_layer(hidden_gene_2)
        output_cna = output_cna_layer(hidden_cna_2)

        autoencoder = Model([input_rna, input_cna, input_gene], [output_rna, output_cna, output_gene])
        autoencoder.compile(loss=['mse', 'mse', 'mse'], optimizer= optimizers.SGD(lr=0.01))

        autoencoder.fit([X_train_rna, X_train_cna, X_train_gene], [X_train_rna, X_train_cna, X_train_gene],
                        epochs=n_multi_epochs, batch_size=32, shuffle=True, verbose=0)

        multi_encoder = Model([input_rna, input_cna, input_gene], combined)

        multi_enc_train = multi_encoder.predict([X_train_rna, X_train_cna, X_train_gene])
        multi_enc_test  = multi_encoder.predict([X_test_rna, X_test_cna, X_test_gene])

        print("before cna")
        df = pandas.DataFrame(X_train_cna)
        df.to_csv('autoEncoderData/cna_data.csv', index=False, header=False)
        df = pandas.DataFrame(X_test_cna)
        df.to_csv('autoEncoderData/test_cna_data.csv', index=False, header=False)
        print("after cna")

        print("before rna")
        df = pandas.DataFrame(X_train_rna)
        df.to_csv('autoEncoderData/rna_data.csv', index=False, header=False)
        df = pandas.DataFrame(X_test_rna)
        df.to_csv('autoEncoderData/test_rna_data.csv', index=False, header=False)
        print("after rna")

        print("before gene")
        df = pandas.DataFrame(X_train_gene)
        df.to_csv('autoEncoderData/gene_data.csv', index=False, header=False)
        df = pandas.DataFrame(X_test_gene)
        df.to_csv('autoEncoderData/test_gene_data.csv', index=False, header=False)
        print("after gene")

        print("before clinical")
        df = pandas.DataFrame(X_train_clinical_data)
        df.to_csv('autoEncoderData/clinical_data.csv', index=False, header=False)
        df = pandas.DataFrame(X_test_clinical_data)
        df.to_csv('autoEncoderData/test_clinical_data.csv', index=False, header=False)
        print("after clinical")

        print("before labels")
        df = pandas.DataFrame(label_train)
        df.to_csv('autoEncoderData/label_train.csv', index=False, header=False)
        df = pandas.DataFrame(label_test)
        df.to_csv('autoEncoderData/label_test.csv', index=False, header=False)
        print("after labels")

        # Evaluate different representations
        entry = []
        # entry.append(run_complex_classifier(multi_enc_train, multi_enc_test))
        # entry.append(run_complex_classifier(X_train_rna, X_test_rna))
        # entry.append(run_complex_classifier(X_train_gene, X_test_gene))
        # entry.append(run_complex_classifier(X_train_cna, X_test_cna))
        # entry.append(run_complex_classifier(np.hstack((X_train_rna, X_train_cna, X_train_gene)), np.hstack((X_test_rna, X_test_cna, X_test_gene))))
        #
        # entry.append(run_simple_classifier(multi_enc_train, multi_enc_test))
        # entry.append(run_simple_classifier(X_train_rna, X_test_rna))
        # entry.append(run_simple_classifier(X_train_gene, X_test_gene))
        # entry.append(run_simple_classifier(X_train_cna, X_test_cna))
        # entry.append(run_simple_classifier(np.hstack((X_train_rna, X_train_cna, X_train_gene)), np.hstack((X_test_rna, X_test_cna, X_test_gene))))

        print(entry)

        data.append(entry)

        return True


#----------------------------------------Classifier-------------------------------------------------------

    def run_complex_classifier(x_train, x_test):

        classifier = GradientBoostingClassifier(n_estimators=100, max_features='log2', random_state=0).fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        return accuracy_score(y_test, y_pred)


    def run_simple_classifier(x_train, x_test):

        classifier = AdaBoostClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        return accuracy_score(y_test, y_pred)

# ----------------------------------------Runner-------------------------------------------------------

    run_multi_encoder(verb=0, n_multi_epochs=200)




# Run for 15 iterations
for i in range(1):
    print("Iteration ", i, "...")
    train_graph()
    print("")
    print("")
    


# Obtain averages
data = np.array(data)

#means, deviations = np.apply_along_axis(func1d=np.mean, axis=0, arr=data), \
 #                   np.apply_along_axis(func1d=np.std, axis=0, arr=data)

#print(means)
#print(deviations)