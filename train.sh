python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.1  --outlier_ratio 0;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.3  --outlier_ratio 0;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.7  --outlier_ratio 0;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.9  --outlier_ratio 0;
 
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  1.0   --outlier_ratio 0;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0.01;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0.02;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0.05;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0.07;
#python siamese_main.py --config cfgs/Vin_siamese.yaml --data_ratio  0.5  --outlier_ratio 0.10;
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.1  --outlier_ratio 0; # lr= 1e-5
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.3  --outlier_ratio 0; # lr= 1e-7
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.49  --outlier_ratio 0; # lr= 1e-7
#python siamese_main.py --config cfgs/RSNA_siamese_without_pretrained.yaml --data_ratio  0.5  --outlier_ratio 0  ; # lr= 1e-7
#python siamese_main.py --config cfgs/Vin_siamese_without_pretrained.yaml  --data_ratio  0.5  --outlier_ratio 0  ;
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.7  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese; # lr= 1e-7
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.9  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese; # lr= 1e-7




#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  1.0   --outlier_ratio 0; # # lr= 1e-7
 
# 
###################### training with barlow as the backbone############
#python siamese_main.py --config cfgs/Vin_ReSSL.yaml --data_ratio  0.5  --outlier_ratio 0 --mode train --k 1 --normalization
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.1  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.3  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.5  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.7  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.9  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.1  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.3  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/Vin_barlow.yaml --epoch 100 --data_ratio  0.5  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/Vin_barlow.yaml --epoch 100 --data_ratio  0.7  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#python siamese_main.py --config cfgs/Vin_barlow.yaml --epoch 100 --data_ratio  0.9  --outlier_ratio 0 --mode train --k 1 --normalization --geometric_mean --with_siamese
#
####  anomaly ratio
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.4906  --outlier_ratio 0.01 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.4906  --outlier_ratio 0.02 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.4906  --outlier_ratio 0.05 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.4906  --outlier_ratio 0.07 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_barlow.yaml --data_ratio  0.4906  --outlier_ratio 0.1 -mode train --k 1 --normalization --geometric_mean --with_siamese;
#
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.5  --outlier_ratio 0.01 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.5  --outlier_ratio 0.02 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.5  --outlier_ratio 0.05 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.5  --outlier_ratio 0.07 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/Vin_barlow.yaml --data_ratio  0.5  --outlier_ratio 0.1 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#

#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.4906  --outlier_ratio 0.01 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.4906  --outlier_ratio 0.02 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.4906  --outlier_ratio 0.05 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_siamese.yaml --data_ratio  0.4906  --outlier_ratio 0.07 --mode train --k 1 --normalization --geometric_mean --with_siamese;
#python siamese_main.py --config cfgs/RSNA_barlow_without_pretrained.yaml --data_ratio  0.5 --mode train --k 1 --normalization --geometric_mean --with_siamese --without_pre_trained;
#python siamese_main.py --config cfgs/Vin_barlow_without_pretrained.yaml  --data_ratio  0.5 --mode train --k 1 --normalization --geometric_mean --with_siamese --without_pre_trained;

 

####################AE model#########################
#python main.py --config cfgs/Vin_AE.yaml --mode b --data_ratio  0.1  --outlier_ratio 0;
#python main.py --config cfgs/Vin_AE.yaml --mode b --data_ratio  0.3  --outlier_ratio 0;
#python main.py --config cfgs/Vin_AE.yaml --mode b --data_ratio  0.5  --outlier_ratio 0;
#python main.py --config cfgs/Vin_AE.yaml --mode b --data_ratio  0.7  --outlier_ratio 0;
#python main.py --config cfgs/Vin_AE.yaml --mode b --data_ratio  0.9  --outlier_ratio 0;
#python main.py --config cfgs/Vin_AE.yaml --mode b --data_ratio  1.0  --outlier_ratio 0;
#python main.py --config cfgs/RSNA_AE.yaml --mode b --data_ratio  0.1  --outlier_ratio 0;
#python main.py --config cfgs/RSNA_AE.yaml --mode b --data_ratio  0.3  --outlier_ratio 0;
#python main.py --config cfgs/RSNA_AE.yaml --mode b --data_ratio  0.5  --outlier_ratio 0;
#python main.py --config cfgs/RSNA_AE.yaml --mode b --data_ratio  0.7  --outlier_ratio 0;
#python main.py --config cfgs/RSNA_AE.yaml --mode b --data_ratio  0.9  --outlier_ratio 0;
#python main.py --config cfgs/RSNA_AE.yaml --mode b --data_ratio  1.0  --outlier_ratio 0;