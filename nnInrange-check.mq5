//+------------------------------------------------------------------+
//|                                                    nnInrange.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"


#define NB_CANDLE 400
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------


#define HANGING_MAN_BODY  0.15
#define HANGING_MAN_HEIGHT  0.75
#define SHOOTING_STAR_HEIGHT  0.25
#define SPINNING_TOP_MIN  0.40
#define SPINNING_TOP_MAX  0.60
#define MARUBOZU_RATE  0.98
#define ENGULFING_FACTOR  1.1
#define MORNING_STAR_PREV2_BODY  0.90
#define MORNING_STAR_PREV_BODY  0.10
#define TWEEZER_BODY  0.15 
#define TWEEZER_HL  0.0001
#define TWEEZER_TOP_BODY  0.40
#define TWEEZER_BOTTOM_BODY  0.60


struct Features_t
{
	double BB_MA_c[];
   	double BB_UP_c[];
   	double BB_LW_c[];

   	double BB_MA_o[];
   	double BB_UP_o[];
   	double BB_LW_o[];

   	double BB_MA_l[];
   	double BB_UP_l[];
   	double BB_LW_l[];

   	double BB_MA_h[];
   	double BB_UP_h[];
   	double BB_LW_h[];
   	
   	double ATR_14[];
   	
   	double EMA_c[];
   	double EMA_h[];
   	double EMA_l[];
   	double EMA_o[];
   	
   	double KeUp_c[];
	double KeUp_h[];
	double KeUp_l[];
	double KeUp_o[]; 
   	
   	double KeLo_c[];
	double KeLo_h[];
	double KeLo_l[];
	double KeLo_o[];
	
	double gains[]; 
	
	double wins_rma[];
	double losses_rma[];
	double RSI_14[];
	double MACD[];
	double SIGNAL_MACD[];
	double HIST[];
	
	double body_lower[];
	double body_upper[];
	double full_range[];
	double body_size[];
	double body_bottom_perc[];
	double body_top_perc[];
	double body_perc[];
	double direction[];
	double low_change[];
	double high_change[];
	double mid_point[];

	double mid_point_prev_2[];
	double body_size_prev[];
	double direction_prev[];
	double direction_prev_2[];
	double body_perc_prev[];
	double body_perc_prev_2[];

	double HANGING_MAN[];
	double SHOOTING_STAR[];
	double SPINNING_TOP[];
	double MARUBOZU[];
	double ENGULFING[];
	double TWEEZER_TOP[];
	double TWEEZER_BOTTOM[];
	double MORNING_STAR[];
	double EVENING_STAR[];
	
	double pivots_l[];
	double pivots_h[];	
	  	
};

int OnInit()
{

	MqlRates bar[];                              
	ArraySetAsSeries(bar,false);//49 is latest candle 48 is perv and so on $FALSE
   	CopyRates(_Symbol,PERIOD_CURRENT,0,NB_CANDLE,bar);
   	
   	Features_t feat;

   	matrix coefs_0 = CMatrixutils_ReadCsv("coefs_0.csv", 283, 200);
	matrix coefs_1 = CMatrixutils_ReadCsv("coefs_1.csv", 200, 200);
	matrix coefs_2 = CMatrixutils_ReadCsv("coefs_2.csv", 200, 3); // satr soton
 	matrix intercepts_0 = CMatrixutils_ReadCsv("intercepts_0.csv", 200, 1); 
 	matrix intercepts_1 = CMatrixutils_ReadCsv("intercepts_1.csv", 200, 1);
 	matrix intercepts_2 = CMatrixutils_ReadCsv("intercepts_2.csv", 3, 1); // 
 
 	
 	//featureExtraction(feat,bar,NB_CANDLE);

	ATRFeature(feat.ATR_14,bar,NB_CANDLE);
   	BollingerBandsFeature(feat,bar,NB_CANDLE);
   	KeltnerChannelsFeature(feat,bar,NB_CANDLE); // TEST THIS!!!!
   	RSIFeature(feat,bar,NB_CANDLE);
   	MACDFeature(feat,bar,NB_CANDLE); // test this
   	candleFeature(feat,bar,NB_CANDLE);  // thest this 

	//int h = FileOpen("feat.csv",FILE_READ|FILE_WRITE|FILE_CSV);
	
	int my_index = NB_CANDLE-3;
	//buildFeaturVectortoFile(feat,my_index);
	//matrix feat_vec = buildFeaturVector(feat,my_index);
	//Print(bar[my_index].time ,"feat_vec = ",feat_vec);
	/* Print(bar[my_index].time,"|o: ",bar[my_index].open
	,"|h: ",bar[my_index].high
	,"|l: ",bar[my_index].low
	,"|c: ",bar[my_index].close); */
	
 	for(int i =200;i <NB_CANDLE;i++) {

	 	matrix feat_vec = buildFeaturVector(feat,i);
		
		//Print(bar[i].time ,"feat_vec = ",feat_vec);
		//int pred = perdict(feat_vec,coefs_0,coefs_1,coefs_2,intercepts_0,intercepts_1,intercepts_2);
		
		matrix l1 = feat_vec.MatMul(coefs_0);
		
		l1 = l1 + intercepts_0.Transpose();
		l1.Activation(l1,AF_RELU);
		
		matrix l2 = l1.MatMul(coefs_1);
		l2 = l2 + intercepts_1.Transpose();
		l2.Activation(l2,AF_RELU);
		
		matrix l3 = l2.MatMul(coefs_2);
		l3 = l3 + intercepts_2.Transpose();
		l3.Activation(l3,AF_SOFTMAX);
		
		int pred = l3.ArgMax(1)[0]-1;
		Print(bar[i].time ,"|\t|",pred);
	}
	//int last_closed_bar = NB_CANDLE-1;
	
	
	
	// if(feat.pivots_l[last_closed_bar-1] == 1.0)
	// {
	// 	// calc NN 
	// 	// if pred == 1 BUYY
	// }

	// if(feat.pivots_h[last_closed_bar-1] == 1.0)
	// {
	// 	// calc NN 
	// 	// if pred == -1 SELL
	// }
  	
	
	
	return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
	   
}

void OnTick()
{

   
}

int perdict(matrix& feat_vec,matrix& coefs_0,matrix& coefs_1,matrix& coefs_2,
		 matrix& intercepts_0, matrix& intercepts_1, matrix& intercepts_2)
{
	matrix l1 = feat_vec.MatMul(coefs_0);
	l1 = l1 + intercepts_0.Transpose();
	l1.Activation(l1,AF_RELU);

	matrix l2 = l1.MatMul(coefs_1);
	l2 = l2 + intercepts_1.Transpose();
	l2.Activation(l2,AF_RELU);
	
	matrix l3 = l2.MatMul(coefs_2);
	l3 = l3 + intercepts_2.Transpose();
	l3.Activation(l3,AF_SOFTMAX);

	int pred = l3.ArgMax(1)[0]-1;
	
	//Print("Cols = \n", l3.Cols()); 
	//Print("Row = \n", l3.Rows());    	
   	//Print("matrix = \n", l3);    	

	return pred;

}
void featureExtraction(Features_t& feat, MqlRates& bar[],int nb_bar)
{
	
}

matrix CMatrixutils_ReadCsv(string file_name,int nb_rows,int nb_cols, string delimiter="," )
{
	matrix mat_ = {};
	mat_.Resize(nb_rows,nb_cols);
	
	int handle = FileOpen(file_name,FILE_READ|FILE_CSV|FILE_ANSI,delimiter);
	//int handle = FileOpen(file_name,FILE_READ,delimiter);
	ResetLastError();
	if(handle == INVALID_HANDLE) {
		printf("Invalid %s handle Error %d ",file_name,GetLastError());
		Print(GetLastError()==0?" TIP | File Might be in use Somewhere else or in another Directory":"");
	}
   	else {
		int column = 0, rows=0;
		
		
		
	 	while(!FileIsEnding(handle)) {
	    	string data = FileReadString(handle);
	    	mat_[rows,column] = (double(data));
         	column++;
	    	if(FileIsLineEnding(handle)){
	    		column = 0;
	    		rows++;
	    	}
	    }	
	
      
    }
    
    FileClose(handle);
 
   return(mat_);
 }
  
  

void candleFeature(Features_t& feat, MqlRates& bar[],int nb_bar)
{
	// todo DO this for all features
	arrayResizeAndInit(feat.body_lower,nb_bar);
	arrayResizeAndInit(feat.body_upper,nb_bar);
	arrayResizeAndInit(feat.full_range,nb_bar);
	arrayResizeAndInit(feat.body_size,nb_bar);
	arrayResizeAndInit(feat.body_bottom_perc,nb_bar);
	arrayResizeAndInit(feat.body_top_perc,nb_bar);
	arrayResizeAndInit(feat.body_perc,nb_bar);
	arrayResizeAndInit(feat.direction,nb_bar);
	arrayResizeAndInit(feat.low_change,nb_bar);
	arrayResizeAndInit(feat.high_change,nb_bar);
	arrayResizeAndInit(feat.mid_point,nb_bar);
	arrayResizeAndInit(feat.mid_point_prev_2,nb_bar);
	arrayResizeAndInit(feat.body_size_prev,nb_bar);
	arrayResizeAndInit(feat.direction_prev,nb_bar);
	arrayResizeAndInit(feat.direction_prev_2,nb_bar);
	arrayResizeAndInit(feat.body_perc_prev,nb_bar);
	arrayResizeAndInit(feat.body_perc_prev_2,nb_bar);
	
	arrayResizeAndInit(feat.HANGING_MAN,nb_bar);
	arrayResizeAndInit(feat.SHOOTING_STAR,nb_bar);
	arrayResizeAndInit(feat.SPINNING_TOP,nb_bar);
	arrayResizeAndInit(feat.MARUBOZU,nb_bar);
	arrayResizeAndInit(feat.ENGULFING,nb_bar);
	arrayResizeAndInit(feat.TWEEZER_TOP,nb_bar);
	arrayResizeAndInit(feat.TWEEZER_BOTTOM,nb_bar);
	arrayResizeAndInit(feat.MORNING_STAR,nb_bar);
	arrayResizeAndInit(feat.EVENING_STAR,nb_bar);

	arrayResizeAndInit(feat.pivots_l,nb_bar);
	arrayResizeAndInit(feat.pivots_h,nb_bar);
	
	double body_size_change[];
	arrayResizeAndInit(body_size_change,nb_bar);


	for(int i =1;i<nb_bar;i++){
		if(bar[i].close <= bar[i].open){
			feat.body_lower[i] = bar[i].close; 
			feat.body_upper[i] = bar[i].open; 
			feat.direction[i] = -1.0;
			feat.body_size[i] = bar[i].open - bar[i].close;
		}
		else {
			feat.body_lower[i] = bar[i].open; 
			feat.body_upper[i] = bar[i].close; 
			feat.direction[i] = +1.0;
			feat.body_size[i] = bar[i].close - bar[i].open;
		}
		
		feat.full_range[i] = bar[i].high - bar[i].low;
		feat.body_bottom_perc[i] = (feat.body_lower[i] - bar[i].low) / feat.full_range[i] ;
		feat.body_top_perc[i] = (feat.body_upper[i] - bar[i].low) / feat.full_range[i] ;
		feat.body_perc[i] = feat.body_size[i] / feat.full_range[i];
		feat.low_change[i] = ( bar[i].low/bar[i-1].low -1 );
		feat.high_change[i] = ( bar[i].high/bar[i-1].high -1 );
		body_size_change[i] = ( feat.body_size[i]/feat.body_size[i-1] -1 ); // this is NOT feature
		feat.mid_point[i] = feat.full_range[i] / 2.0 + bar[i].low;
		feat.body_size_prev[i] = feat.body_size[i-1];
		feat.direction_prev[i] = feat.direction[i-1];
		feat.body_perc_prev[i] = feat.body_perc[i-1];
		if(i>=2){
			feat.mid_point_prev_2[i] = feat.mid_point[i-2];
			feat.direction_prev_2[i] = feat.direction[i-2];
			feat.body_perc_prev_2[i] = feat.body_perc[i-2];

		}

		if(	feat.body_bottom_perc[i] > HANGING_MAN_HEIGHT &&
		 	feat.body_perc[i] < HANGING_MAN_BODY )
            feat.HANGING_MAN[i] = 1.0;

		if(	feat.body_top_perc[i] < SHOOTING_STAR_HEIGHT &&
			feat.body_perc[i] < HANGING_MAN_BODY )
			feat.SHOOTING_STAR[i] = 1.0;

		if( feat.body_top_perc[i] < SPINNING_TOP_MAX &&
        	feat.body_bottom_perc[i] > SPINNING_TOP_MIN &&
            feat.body_perc[i] < HANGING_MAN_BODY )
			feat.SPINNING_TOP[i] = 1.0;

		if(feat.body_perc[i] > MARUBOZU_RATE)
			feat.MARUBOZU[i] = 1.0;

		if(	feat.direction[i] != feat.direction_prev[i] &&
        	feat.body_size[i] > feat.body_size_prev[i] * ENGULFING_FACTOR)
			feat.ENGULFING[i] = 1.0;

		if( MathAbs(body_size_change[i]) < TWEEZER_BODY &&
        	feat.direction[i] == -1 && feat.direction[i] != feat.direction_prev[i] &&
            MathAbs(feat.low_change[i]) < TWEEZER_HL && MathAbs(feat.high_change[i]) < TWEEZER_HL &&
        	feat.body_top_perc[i] < TWEEZER_TOP_BODY )
			feat.TWEEZER_TOP[i] = 1.0;

		if(	MathAbs(body_size_change[i]) < TWEEZER_BODY &&
        	feat.direction[i] == 1 && feat.direction[i] != feat.direction_prev[i] &&
            MathAbs(feat.low_change[i]) < TWEEZER_HL && MathAbs(feat.high_change[i]) < TWEEZER_HL &&
            feat.body_bottom_perc[i] > TWEEZER_BOTTOM_BODY)
			feat.TWEEZER_BOTTOM[i] = 1.0;


		if(i>=2){
			if( feat.body_perc_prev_2[i] > MORNING_STAR_PREV2_BODY &&
				feat.body_perc_prev[i] < MORNING_STAR_PREV_BODY &&
				feat.direction[i] == 1 && feat.direction_prev_2[i] != 1 &&
				bar[i].close > feat.mid_point_prev_2[i])
				feat.MORNING_STAR[i] = 1.0;
			
				if( feat.body_perc_prev_2[i] > MORNING_STAR_PREV2_BODY &&
					feat.body_perc_prev[i] < MORNING_STAR_PREV_BODY &&
					feat.direction[i] == -1 && feat.direction_prev_2[i] != -1 &&
					bar[i].close < feat.mid_point_prev_2[i])
					feat.EVENING_STAR[i] = 1.0;

		if(	(bar[i-2].low > bar[i-1].low) && (bar[i].low > bar[i-1].low) )
    		feat.pivots_l[i-1] = 1.0;

		if(	(bar[i-2].high < bar[i-1].high) && (bar[i].high < bar[i-1].high) )
    		feat.pivots_h[i-1] = 1.0;
				
		}
		
	}
}

void MACDFeature(Features_t& feat, MqlRates& bar[],int nb_bar,int n_slow=26, int n_fast=12, int n_signal=9)
{
	ArrayResize(feat.MACD,nb_bar);
	ArrayInitialize(feat.MACD,0.0);
	ArrayResize(feat.SIGNAL_MACD,nb_bar);
	ArrayInitialize(feat.SIGNAL_MACD,0.0);
	ArrayResize(feat.HIST,nb_bar);
	ArrayInitialize(feat.HIST,0.0);
	
	double alpha_slow = 1.0/(double)n_slow;
	double alpha_fast = 1.0/(double)n_fast;
	double alpha_signal = 1.0/(double)n_signal;	
	
	double ema_long[];
	ArrayResize(ema_long,nb_bar);
	ArrayInitialize(ema_long,0.0);
	
	double ema_short[];
	ArrayResize(ema_short,nb_bar);
	ArrayInitialize(ema_short,0.0);
	
	
	ema_long[0] = bar[0].close;
	ema_short[0] = bar[0].close;
	for(int i =1;i<nb_bar;i++){
		ema_long[i] = bar[i].close*alpha_slow + ema_long[i-1]*(1 - alpha_slow);
		ema_short[i] = bar[i].close*alpha_fast + ema_short[i-1]*(1 - alpha_fast);
		
		feat.MACD[i] = ema_short[i] - ema_long[i];
		if(i == 1)
			feat.SIGNAL_MACD[i] = feat.MACD[i];
		else {
			feat.SIGNAL_MACD[i] = feat.MACD[i]*alpha_signal + feat.SIGNAL_MACD[i-1]*(1-alpha_signal);
			feat.HIST[i] = feat.MACD[i]-feat.SIGNAL_MACD[i];
		}
		
	}
	
}

void RSIFeature(Features_t& feat, MqlRates& bar[],int nb_bar,int window = 14)
{
	double alpha = 1.0/(double)window;
	ArrayResize(feat.gains,nb_bar);
	ArrayInitialize(feat.gains,0.0);
	
	double wins[];
	ArrayResize(wins,nb_bar);
	ArrayInitialize(wins,0.0);
	
	double losses[];
	ArrayResize(losses,nb_bar);
	ArrayInitialize(losses,0.0);
	
	ArrayResize(feat.wins_rma,nb_bar);
	ArrayInitialize(feat.wins_rma,0.0);
	
	ArrayResize(feat.losses_rma,nb_bar);
	ArrayInitialize(feat.losses_rma,0.0);
	
	ArrayResize(feat.RSI_14,nb_bar);
	ArrayInitialize(feat.RSI_14,0.0);
	
	for(int i =1;i<nb_bar;i++){
		feat.gains[i] = bar[i].close - bar[i-1].close;
		if(feat.gains[i]>0 ) 
			wins[i] = feat.gains[i];
		else 
			losses[i] = -feat.gains[i];
			
		feat.wins_rma[i] = wins[i]*alpha + feat.wins_rma[i-1]*(1-alpha);
		feat.losses_rma[i] = losses[i]*alpha + feat.losses_rma[i-1]*(1-alpha);
		double rs = feat.wins_rma[i] / feat.losses_rma[i];
		feat.RSI_14[i] = 1.0 - (1.0 / (1.0 + rs));
	}
	
	
}

void KeltnerChannelsFeature(Features_t& feat, MqlRates& bar[],int nb_bar,int n_ema=20, int n_atr=10)
{
	double EMA[];
	arrayResizeAndInit(EMA,nb_bar);
	double ATR_10[];
	arrayResizeAndInit(ATR_10,nb_bar);
	
	ATRFeature(ATR_10,bar,nb_bar,n_atr);
	
	arrayResizeAndInit(feat.EMA_c,nb_bar);
	arrayResizeAndInit(feat.EMA_h,nb_bar);
	arrayResizeAndInit(feat.EMA_l,nb_bar);
	arrayResizeAndInit(feat.EMA_o,nb_bar);
	
	arrayResizeAndInit(feat.KeUp_c,nb_bar);
	arrayResizeAndInit(feat.KeUp_h,nb_bar);
	arrayResizeAndInit(feat.KeUp_l,nb_bar);
	arrayResizeAndInit(feat.KeUp_o,nb_bar);
	
	arrayResizeAndInit(feat.KeLo_c,nb_bar);
	arrayResizeAndInit(feat.KeLo_h,nb_bar);
	arrayResizeAndInit(feat.KeLo_l,nb_bar);
	arrayResizeAndInit(feat.KeLo_o,nb_bar);
	
	double alpha = 2.0 / ( (double )n_ema + 1.0);
	EMA[0] = bar[0].close;
	for(int i=1;i<nb_bar;i++){
		EMA[i] = bar[i].close*alpha + EMA[i-1]*(1 - alpha);
		
		feat.EMA_c[i] = EMA[i]-bar[i].close;
		feat.EMA_h[i] = EMA[i]-bar[i].high;
		feat.EMA_l[i] = EMA[i]-bar[i].low;
		feat.EMA_o[i] = EMA[i]-bar[i].open;
		
		double KeUp = ATR_10[i]*2.0 + EMA[i];
		double KeLo = EMA[i] - 2.0*ATR_10[i];
		
		feat.KeUp_c[i] = KeUp - bar[i].close;
		feat.KeUp_h[i] = KeUp - bar[i].high;
		feat.KeUp_l[i] = KeUp - bar[i].low;
		feat.KeUp_o[i] = KeUp - bar[i].open;
		
		feat.KeLo_c[i] = KeLo - bar[i].close;
		feat.KeLo_h[i] = KeLo - bar[i].high;
		feat.KeLo_l[i] = KeLo - bar[i].low;
		feat.KeLo_o[i] = KeLo - bar[i].open;
		
	}
	    
	    
	return;
	
	
}
void ATRFeature(double &ATR[], MqlRates& bar[], int nb_bar, int window = 14)
{
	
	arrayResizeAndInit(ATR,nb_bar);
	
	double tr[];
	arrayResizeAndInit(tr,nb_bar);
	
	for(int i =1;i<nb_bar;i++){
		double prev_c = bar[i-1].close;
		double tr1 = bar[i].high - bar[i].low;
		double tr2 = bar[i].high - prev_c;
		double tr3 = prev_c - bar[i].low;
		
		if (tr1 >= tr2 && tr1 >= tr3 )
			tr[i] = tr1;
		
		if (tr2 >= tr1 && tr2 >= tr3 )
			tr[i] = tr2;
		
		if (tr3 >= tr2 && tr3 >= tr1 )
			tr[i] = tr3;
			
		if( i > window ) {
   		 	double ATR_SUM_TEMP = 0.0;
   			
   			for(int j=0;j<window;j++)
   				ATR_SUM_TEMP += (tr[i-j]);
   			
   			ATR[i] = ATR_SUM_TEMP/(double)window;
		}
	}
	
	
}

void BollingerBandsFeature(Features_t& feat, MqlRates& bar[], int nb_bar, int window = 20, double s = 2.0)
{
	arrayResizeAndInit(feat.BB_MA_c,nb_bar);
	arrayResizeAndInit(feat.BB_UP_c,nb_bar); 
	arrayResizeAndInit(feat.BB_LW_c,nb_bar);	
	arrayResizeAndInit(feat.BB_MA_o,nb_bar);
	arrayResizeAndInit(feat.BB_UP_o,nb_bar);
	arrayResizeAndInit(feat.BB_LW_o,nb_bar);
	arrayResizeAndInit(feat.BB_MA_l,nb_bar);
	arrayResizeAndInit(feat.BB_UP_l,nb_bar);
	arrayResizeAndInit(feat.BB_LW_l,nb_bar);
	arrayResizeAndInit(feat.BB_MA_h,nb_bar);
	arrayResizeAndInit(feat.BB_UP_h,nb_bar);
	arrayResizeAndInit(feat.BB_LW_h,nb_bar);
	// free up these ...
	double typical_p[];
	arrayResizeAndInit(typical_p,nb_bar);

	double BB_MA[];
	arrayResizeAndInit(BB_MA,nb_bar);
   	
   	double stddev[];
   	arrayResizeAndInit(stddev,nb_bar);
 
 
	for(int i=0;i<nb_bar;i++) {
   		typical_p[i] = (bar[i].close + bar[i].low + bar[i].high)/3;
   	}
   	for(int i=0;i<nb_bar;i++) {
   		if( i > window ) {
   		 	double BB_MA_SUM_TEMP = 0.0;
   			for(int j=0;j<window;j++)
   				BB_MA_SUM_TEMP += (typical_p[i-j]);
   				 
   			BB_MA[i] = BB_MA_SUM_TEMP/(double)window;
   		}
   	}
   	
   	for(int i=0;i<nb_bar;i++) {
   		double std_temp = 0.0;
   		if( i > window ) {
   			for(int j=0;j<window;j++)
   				std_temp += (typical_p[i-j]-BB_MA[i])*(typical_p[i-j]-BB_MA[i]);
   			stddev[i] = sqrt(std_temp/(double)(window-1) );
   		}
   	}
   	
   	for(int i=0;i<nb_bar;i++) {
   		double BB_UP = BB_MA[i] + s*stddev[i];
 		double BB_LW = BB_MA[i] - s*stddev[i];
 		
      	feat.BB_MA_c[i] = BB_MA[i] - bar[i].close;
   		feat.BB_UP_c[i] = BB_UP - bar[i].close;
   		feat.BB_LW_c[i] = BB_LW - bar[i].close;
   		
   		feat.BB_MA_h[i] = BB_MA[i] - bar[i].high;
   		feat.BB_UP_h[i] = BB_UP - bar[i].high;
   		feat.BB_LW_h[i] = BB_LW - bar[i].high;
   		
   		feat.BB_MA_l[i] = BB_MA[i] - bar[i].low;
   		feat.BB_UP_l[i] = BB_UP - bar[i].low;
   		feat.BB_LW_l[i] = BB_LW - bar[i].low;
   		
   		feat.BB_MA_o[i] = BB_MA[i] - bar[i].open;
   		feat.BB_UP_o[i] = BB_UP - bar[i].open;
   		feat.BB_LW_o[i] = BB_LW - bar[i].open;
   	}
   	
   	return;
}

void arrayResizeAndInit(double& arr[], int size, double val = 0.0)
{
	ArrayResize(arr,size);  	
	ArrayInitialize(arr,val);
}



matrix buildFeaturVector(Features_t & feat, int bar_idx)
{
	matrix feat_vec;
 	feat_vec.Resize(1,283);
 	
 	int idx = 0;

	feat_vec[0,idx] = feat.pivots_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.pivots_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.pivots_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.pivots_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.pivots_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.pivots_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.pivots_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.pivots_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_MA_c[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_MA_c[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_MA_c[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_MA_c[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_MA_c[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_UP_c[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_UP_c[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_UP_c[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_UP_c[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_UP_c[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_LW_c[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_LW_c[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_LW_c[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_LW_c[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_LW_c[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.EMA_c[bar_idx]; idx++;
	feat_vec[0,idx] = feat.EMA_c[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.EMA_c[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.EMA_c[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.EMA_c[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeUp_c[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeUp_c[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeUp_c[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeUp_c[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeUp_c[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeLo_c[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeLo_c[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeLo_c[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeLo_c[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeLo_c[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_MA_o[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_MA_o[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_MA_o[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_MA_o[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_MA_o[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_UP_o[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_UP_o[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_UP_o[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_UP_o[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_UP_o[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_LW_o[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_LW_o[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_LW_o[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_LW_o[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_LW_o[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.EMA_o[bar_idx]; idx++;
	feat_vec[0,idx] = feat.EMA_o[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.EMA_o[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.EMA_o[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.EMA_o[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeUp_o[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeUp_o[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeUp_o[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeUp_o[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeUp_o[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeLo_o[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeLo_o[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeLo_o[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeLo_o[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeLo_o[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_MA_l[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_MA_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_MA_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_MA_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_MA_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_UP_l[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_UP_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_UP_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_UP_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_UP_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_LW_l[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_LW_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_LW_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_LW_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_LW_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.EMA_l[bar_idx]; idx++;
	feat_vec[0,idx] = feat.EMA_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.EMA_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.EMA_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.EMA_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeUp_l[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeUp_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeUp_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeUp_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeUp_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeLo_l[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeLo_l[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeLo_l[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeLo_l[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeLo_l[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_MA_h[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_MA_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_MA_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_MA_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_MA_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_UP_h[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_UP_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_UP_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_UP_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_UP_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.BB_LW_h[bar_idx]; idx++;
	feat_vec[0,idx] = feat.BB_LW_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.BB_LW_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.BB_LW_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.BB_LW_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.EMA_h[bar_idx]; idx++;
	feat_vec[0,idx] = feat.EMA_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.EMA_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.EMA_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.EMA_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeUp_h[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeUp_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeUp_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeUp_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeUp_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.KeLo_h[bar_idx]; idx++;
	feat_vec[0,idx] = feat.KeLo_h[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.KeLo_h[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.KeLo_h[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.KeLo_h[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.ATR_14[bar_idx]; idx++;
	feat_vec[0,idx] = feat.ATR_14[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.ATR_14[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.ATR_14[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.ATR_14[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.gains[bar_idx]; idx++;
	feat_vec[0,idx] = feat.gains[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.gains[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.gains[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.gains[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.wins_rma[bar_idx]; idx++;
	feat_vec[0,idx] = feat.wins_rma[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.wins_rma[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.wins_rma[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.wins_rma[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.losses_rma[bar_idx]; idx++;
	feat_vec[0,idx] = feat.losses_rma[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.losses_rma[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.losses_rma[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.losses_rma[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.RSI_14[bar_idx]; idx++;
	feat_vec[0,idx] = feat.RSI_14[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.RSI_14[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.RSI_14[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.RSI_14[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.full_range[bar_idx]; idx++;
	feat_vec[0,idx] = feat.full_range[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.full_range[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.full_range[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.full_range[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_lower[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_lower[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_lower[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_lower[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_lower[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_upper[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_upper[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_upper[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_upper[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_upper[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_bottom_perc[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_bottom_perc[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_bottom_perc[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_bottom_perc[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_bottom_perc[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_top_perc[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_top_perc[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_top_perc[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_top_perc[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_top_perc[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_perc[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_perc[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_perc[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_perc[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_perc[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.direction[bar_idx]; idx++;
	feat_vec[0,idx] = feat.direction[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.direction[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.direction[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.direction[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_size[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_size[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_size[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_size[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_size[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.low_change[bar_idx]; idx++;
	feat_vec[0,idx] = feat.low_change[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.low_change[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.low_change[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.low_change[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.high_change[bar_idx]; idx++;
	feat_vec[0,idx] = feat.high_change[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.high_change[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.high_change[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.high_change[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.mid_point[bar_idx]; idx++;
	feat_vec[0,idx] = feat.mid_point[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.mid_point[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.mid_point[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.mid_point[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.mid_point_prev_2[bar_idx]; idx++;
	feat_vec[0,idx] = feat.mid_point_prev_2[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.mid_point_prev_2[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.mid_point_prev_2[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.mid_point_prev_2[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_size_prev[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_size_prev[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_size_prev[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_size_prev[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_size_prev[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.direction_prev[bar_idx]; idx++;
	feat_vec[0,idx] = feat.direction_prev[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.direction_prev[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.direction_prev[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.direction_prev[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.direction_prev_2[bar_idx]; idx++;
	feat_vec[0,idx] = feat.direction_prev_2[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.direction_prev_2[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.direction_prev_2[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.direction_prev_2[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev_2[bar_idx]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev_2[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev_2[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev_2[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.body_perc_prev_2[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.HANGING_MAN[bar_idx]; idx++;
	feat_vec[0,idx] = feat.HANGING_MAN[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.HANGING_MAN[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.HANGING_MAN[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.HANGING_MAN[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.SHOOTING_STAR[bar_idx]; idx++;
	feat_vec[0,idx] = feat.SHOOTING_STAR[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.SHOOTING_STAR[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.SHOOTING_STAR[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.SHOOTING_STAR[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.SPINNING_TOP[bar_idx]; idx++;
	feat_vec[0,idx] = feat.SPINNING_TOP[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.SPINNING_TOP[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.SPINNING_TOP[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.SPINNING_TOP[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.MARUBOZU[bar_idx]; idx++;
	feat_vec[0,idx] = feat.MARUBOZU[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.MARUBOZU[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.MARUBOZU[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.MARUBOZU[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.ENGULFING[bar_idx]; idx++;
	feat_vec[0,idx] = feat.ENGULFING[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.ENGULFING[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.ENGULFING[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.ENGULFING[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_TOP[bar_idx]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_TOP[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_TOP[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_TOP[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_TOP[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_BOTTOM[bar_idx]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_BOTTOM[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_BOTTOM[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_BOTTOM[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.TWEEZER_BOTTOM[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.MORNING_STAR[bar_idx]; idx++;
	feat_vec[0,idx] = feat.MORNING_STAR[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.MORNING_STAR[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.MORNING_STAR[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.MORNING_STAR[bar_idx-4]; idx++;
	feat_vec[0,idx] = feat.EVENING_STAR[bar_idx]; idx++;
	feat_vec[0,idx] = feat.EVENING_STAR[bar_idx-1]; idx++;
	feat_vec[0,idx] = feat.EVENING_STAR[bar_idx-2]; idx++;
	feat_vec[0,idx] = feat.EVENING_STAR[bar_idx-3]; idx++;
	feat_vec[0,idx] = feat.EVENING_STAR[bar_idx-4]; idx++;


	return feat_vec;
}



void buildFeaturVectortoFile(Features_t & feat, int bar_idx)
{
	// matrix feat_vec;
 	// feat_vec.Resize(1,283);
 	
 	int idx = 0;
    int h = FileOpen("feat.csv",FILE_READ|FILE_WRITE|FILE_CSV);

FileWrite(h,feat.pivots_h[bar_idx-1]); idx++;
FileWrite(h,feat.pivots_l[bar_idx-1]); idx++;
FileWrite(h,feat.pivots_h[bar_idx-2]); idx++;
FileWrite(h,feat.pivots_l[bar_idx-2]); idx++;
FileWrite(h,feat.pivots_h[bar_idx-3]); idx++;
FileWrite(h,feat.pivots_l[bar_idx-3]); idx++;
FileWrite(h,feat.pivots_h[bar_idx-4]); idx++;
FileWrite(h,feat.pivots_l[bar_idx-4]); idx++;
FileWrite(h,feat.BB_MA_c[bar_idx]); idx++;
FileWrite(h,feat.BB_MA_c[bar_idx-1]); idx++;
FileWrite(h,feat.BB_MA_c[bar_idx-2]); idx++;
FileWrite(h,feat.BB_MA_c[bar_idx-3]); idx++;
FileWrite(h,feat.BB_MA_c[bar_idx-4]); idx++;
FileWrite(h,feat.BB_UP_c[bar_idx]); idx++;
FileWrite(h,feat.BB_UP_c[bar_idx-1]); idx++;
FileWrite(h,feat.BB_UP_c[bar_idx-2]); idx++;
FileWrite(h,feat.BB_UP_c[bar_idx-3]); idx++;
FileWrite(h,feat.BB_UP_c[bar_idx-4]); idx++;
FileWrite(h,feat.BB_LW_c[bar_idx]); idx++;
FileWrite(h,feat.BB_LW_c[bar_idx-1]); idx++;
FileWrite(h,feat.BB_LW_c[bar_idx-2]); idx++;
FileWrite(h,feat.BB_LW_c[bar_idx-3]); idx++;
FileWrite(h,feat.BB_LW_c[bar_idx-4]); idx++;
FileWrite(h,feat.EMA_c[bar_idx]); idx++;
FileWrite(h,feat.EMA_c[bar_idx-1]); idx++;
FileWrite(h,feat.EMA_c[bar_idx-2]); idx++;
FileWrite(h,feat.EMA_c[bar_idx-3]); idx++;
FileWrite(h,feat.EMA_c[bar_idx-4]); idx++;
FileWrite(h,feat.KeUp_c[bar_idx]); idx++;
FileWrite(h,feat.KeUp_c[bar_idx-1]); idx++;
FileWrite(h,feat.KeUp_c[bar_idx-2]); idx++;
FileWrite(h,feat.KeUp_c[bar_idx-3]); idx++;
FileWrite(h,feat.KeUp_c[bar_idx-4]); idx++;
FileWrite(h,feat.KeLo_c[bar_idx]); idx++;
FileWrite(h,feat.KeLo_c[bar_idx-1]); idx++;
FileWrite(h,feat.KeLo_c[bar_idx-2]); idx++;
FileWrite(h,feat.KeLo_c[bar_idx-3]); idx++;
FileWrite(h,feat.KeLo_c[bar_idx-4]); idx++;
FileWrite(h,feat.BB_MA_o[bar_idx]); idx++;
FileWrite(h,feat.BB_MA_o[bar_idx-1]); idx++;
FileWrite(h,feat.BB_MA_o[bar_idx-2]); idx++;
FileWrite(h,feat.BB_MA_o[bar_idx-3]); idx++;
FileWrite(h,feat.BB_MA_o[bar_idx-4]); idx++;
FileWrite(h,feat.BB_UP_o[bar_idx]); idx++;
FileWrite(h,feat.BB_UP_o[bar_idx-1]); idx++;
FileWrite(h,feat.BB_UP_o[bar_idx-2]); idx++;
FileWrite(h,feat.BB_UP_o[bar_idx-3]); idx++;
FileWrite(h,feat.BB_UP_o[bar_idx-4]); idx++;
FileWrite(h,feat.BB_LW_o[bar_idx]); idx++;
FileWrite(h,feat.BB_LW_o[bar_idx-1]); idx++;
FileWrite(h,feat.BB_LW_o[bar_idx-2]); idx++;
FileWrite(h,feat.BB_LW_o[bar_idx-3]); idx++;
FileWrite(h,feat.BB_LW_o[bar_idx-4]); idx++;
FileWrite(h,feat.EMA_o[bar_idx]); idx++;
FileWrite(h,feat.EMA_o[bar_idx-1]); idx++;
FileWrite(h,feat.EMA_o[bar_idx-2]); idx++;
FileWrite(h,feat.EMA_o[bar_idx-3]); idx++;
FileWrite(h,feat.EMA_o[bar_idx-4]); idx++;
FileWrite(h,feat.KeUp_o[bar_idx]); idx++;
FileWrite(h,feat.KeUp_o[bar_idx-1]); idx++;
FileWrite(h,feat.KeUp_o[bar_idx-2]); idx++;
FileWrite(h,feat.KeUp_o[bar_idx-3]); idx++;
FileWrite(h,feat.KeUp_o[bar_idx-4]); idx++;
FileWrite(h,feat.KeLo_o[bar_idx]); idx++;
FileWrite(h,feat.KeLo_o[bar_idx-1]); idx++;
FileWrite(h,feat.KeLo_o[bar_idx-2]); idx++;
FileWrite(h,feat.KeLo_o[bar_idx-3]); idx++;
FileWrite(h,feat.KeLo_o[bar_idx-4]); idx++;
FileWrite(h,feat.BB_MA_l[bar_idx]); idx++;
FileWrite(h,feat.BB_MA_l[bar_idx-1]); idx++;
FileWrite(h,feat.BB_MA_l[bar_idx-2]); idx++;
FileWrite(h,feat.BB_MA_l[bar_idx-3]); idx++;
FileWrite(h,feat.BB_MA_l[bar_idx-4]); idx++;
FileWrite(h,feat.BB_UP_l[bar_idx]); idx++;
FileWrite(h,feat.BB_UP_l[bar_idx-1]); idx++;
FileWrite(h,feat.BB_UP_l[bar_idx-2]); idx++;
FileWrite(h,feat.BB_UP_l[bar_idx-3]); idx++;
FileWrite(h,feat.BB_UP_l[bar_idx-4]); idx++;
FileWrite(h,feat.BB_LW_l[bar_idx]); idx++;
FileWrite(h,feat.BB_LW_l[bar_idx-1]); idx++;
FileWrite(h,feat.BB_LW_l[bar_idx-2]); idx++;
FileWrite(h,feat.BB_LW_l[bar_idx-3]); idx++;
FileWrite(h,feat.BB_LW_l[bar_idx-4]); idx++;
FileWrite(h,feat.EMA_l[bar_idx]); idx++;
FileWrite(h,feat.EMA_l[bar_idx-1]); idx++;
FileWrite(h,feat.EMA_l[bar_idx-2]); idx++;
FileWrite(h,feat.EMA_l[bar_idx-3]); idx++;
FileWrite(h,feat.EMA_l[bar_idx-4]); idx++;
FileWrite(h,feat.KeUp_l[bar_idx]); idx++;
FileWrite(h,feat.KeUp_l[bar_idx-1]); idx++;
FileWrite(h,feat.KeUp_l[bar_idx-2]); idx++;
FileWrite(h,feat.KeUp_l[bar_idx-3]); idx++;
FileWrite(h,feat.KeUp_l[bar_idx-4]); idx++;
FileWrite(h,feat.KeLo_l[bar_idx]); idx++;
FileWrite(h,feat.KeLo_l[bar_idx-1]); idx++;
FileWrite(h,feat.KeLo_l[bar_idx-2]); idx++;
FileWrite(h,feat.KeLo_l[bar_idx-3]); idx++;
FileWrite(h,feat.KeLo_l[bar_idx-4]); idx++;
FileWrite(h,feat.BB_MA_h[bar_idx]); idx++;
FileWrite(h,feat.BB_MA_h[bar_idx-1]); idx++;
FileWrite(h,feat.BB_MA_h[bar_idx-2]); idx++;
FileWrite(h,feat.BB_MA_h[bar_idx-3]); idx++;
FileWrite(h,feat.BB_MA_h[bar_idx-4]); idx++;
FileWrite(h,feat.BB_UP_h[bar_idx]); idx++;
FileWrite(h,feat.BB_UP_h[bar_idx-1]); idx++;
FileWrite(h,feat.BB_UP_h[bar_idx-2]); idx++;
FileWrite(h,feat.BB_UP_h[bar_idx-3]); idx++;
FileWrite(h,feat.BB_UP_h[bar_idx-4]); idx++;
FileWrite(h,feat.BB_LW_h[bar_idx]); idx++;
FileWrite(h,feat.BB_LW_h[bar_idx-1]); idx++;
FileWrite(h,feat.BB_LW_h[bar_idx-2]); idx++;
FileWrite(h,feat.BB_LW_h[bar_idx-3]); idx++;
FileWrite(h,feat.BB_LW_h[bar_idx-4]); idx++;
FileWrite(h,feat.EMA_h[bar_idx]); idx++;
FileWrite(h,feat.EMA_h[bar_idx-1]); idx++;
FileWrite(h,feat.EMA_h[bar_idx-2]); idx++;
FileWrite(h,feat.EMA_h[bar_idx-3]); idx++;
FileWrite(h,feat.EMA_h[bar_idx-4]); idx++;
FileWrite(h,feat.KeUp_h[bar_idx]); idx++;
FileWrite(h,feat.KeUp_h[bar_idx-1]); idx++;
FileWrite(h,feat.KeUp_h[bar_idx-2]); idx++;
FileWrite(h,feat.KeUp_h[bar_idx-3]); idx++;
FileWrite(h,feat.KeUp_h[bar_idx-4]); idx++;
FileWrite(h,feat.KeLo_h[bar_idx]); idx++;
FileWrite(h,feat.KeLo_h[bar_idx-1]); idx++;
FileWrite(h,feat.KeLo_h[bar_idx-2]); idx++;
FileWrite(h,feat.KeLo_h[bar_idx-3]); idx++;
FileWrite(h,feat.KeLo_h[bar_idx-4]); idx++;
FileWrite(h,feat.ATR_14[bar_idx]); idx++;
FileWrite(h,feat.ATR_14[bar_idx-1]); idx++;
FileWrite(h,feat.ATR_14[bar_idx-2]); idx++;
FileWrite(h,feat.ATR_14[bar_idx-3]); idx++;
FileWrite(h,feat.ATR_14[bar_idx-4]); idx++;
FileWrite(h,feat.gains[bar_idx]); idx++;
FileWrite(h,feat.gains[bar_idx-1]); idx++;
FileWrite(h,feat.gains[bar_idx-2]); idx++;
FileWrite(h,feat.gains[bar_idx-3]); idx++;
FileWrite(h,feat.gains[bar_idx-4]); idx++;
FileWrite(h,feat.wins_rma[bar_idx]); idx++;
FileWrite(h,feat.wins_rma[bar_idx-1]); idx++;
FileWrite(h,feat.wins_rma[bar_idx-2]); idx++;
FileWrite(h,feat.wins_rma[bar_idx-3]); idx++;
FileWrite(h,feat.wins_rma[bar_idx-4]); idx++;
FileWrite(h,feat.losses_rma[bar_idx]); idx++;
FileWrite(h,feat.losses_rma[bar_idx-1]); idx++;
FileWrite(h,feat.losses_rma[bar_idx-2]); idx++;
FileWrite(h,feat.losses_rma[bar_idx-3]); idx++;
FileWrite(h,feat.losses_rma[bar_idx-4]); idx++;
FileWrite(h,feat.RSI_14[bar_idx]); idx++;
FileWrite(h,feat.RSI_14[bar_idx-1]); idx++;
FileWrite(h,feat.RSI_14[bar_idx-2]); idx++;
FileWrite(h,feat.RSI_14[bar_idx-3]); idx++;
FileWrite(h,feat.RSI_14[bar_idx-4]); idx++;
FileWrite(h,feat.full_range[bar_idx]); idx++;
FileWrite(h,feat.full_range[bar_idx-1]); idx++;
FileWrite(h,feat.full_range[bar_idx-2]); idx++;
FileWrite(h,feat.full_range[bar_idx-3]); idx++;
FileWrite(h,feat.full_range[bar_idx-4]); idx++;
FileWrite(h,feat.body_lower[bar_idx]); idx++;
FileWrite(h,feat.body_lower[bar_idx-1]); idx++;
FileWrite(h,feat.body_lower[bar_idx-2]); idx++;
FileWrite(h,feat.body_lower[bar_idx-3]); idx++;
FileWrite(h,feat.body_lower[bar_idx-4]); idx++;
FileWrite(h,feat.body_upper[bar_idx]); idx++;
FileWrite(h,feat.body_upper[bar_idx-1]); idx++;
FileWrite(h,feat.body_upper[bar_idx-2]); idx++;
FileWrite(h,feat.body_upper[bar_idx-3]); idx++;
FileWrite(h,feat.body_upper[bar_idx-4]); idx++;
FileWrite(h,feat.body_bottom_perc[bar_idx]); idx++;
FileWrite(h,feat.body_bottom_perc[bar_idx-1]); idx++;
FileWrite(h,feat.body_bottom_perc[bar_idx-2]); idx++;
FileWrite(h,feat.body_bottom_perc[bar_idx-3]); idx++;
FileWrite(h,feat.body_bottom_perc[bar_idx-4]); idx++;
FileWrite(h,feat.body_top_perc[bar_idx]); idx++;
FileWrite(h,feat.body_top_perc[bar_idx-1]); idx++;
FileWrite(h,feat.body_top_perc[bar_idx-2]); idx++;
FileWrite(h,feat.body_top_perc[bar_idx-3]); idx++;
FileWrite(h,feat.body_top_perc[bar_idx-4]); idx++;
FileWrite(h,feat.body_perc[bar_idx]); idx++;
FileWrite(h,feat.body_perc[bar_idx-1]); idx++;
FileWrite(h,feat.body_perc[bar_idx-2]); idx++;
FileWrite(h,feat.body_perc[bar_idx-3]); idx++;
FileWrite(h,feat.body_perc[bar_idx-4]); idx++;
FileWrite(h,feat.direction[bar_idx]); idx++;
FileWrite(h,feat.direction[bar_idx-1]); idx++;
FileWrite(h,feat.direction[bar_idx-2]); idx++;
FileWrite(h,feat.direction[bar_idx-3]); idx++;
FileWrite(h,feat.direction[bar_idx-4]); idx++;
FileWrite(h,feat.body_size[bar_idx]); idx++;
FileWrite(h,feat.body_size[bar_idx-1]); idx++;
FileWrite(h,feat.body_size[bar_idx-2]); idx++;
FileWrite(h,feat.body_size[bar_idx-3]); idx++;
FileWrite(h,feat.body_size[bar_idx-4]); idx++;
FileWrite(h,feat.low_change[bar_idx]); idx++;
FileWrite(h,feat.low_change[bar_idx-1]); idx++;
FileWrite(h,feat.low_change[bar_idx-2]); idx++;
FileWrite(h,feat.low_change[bar_idx-3]); idx++;
FileWrite(h,feat.low_change[bar_idx-4]); idx++;
FileWrite(h,feat.high_change[bar_idx]); idx++;
FileWrite(h,feat.high_change[bar_idx-1]); idx++;
FileWrite(h,feat.high_change[bar_idx-2]); idx++;
FileWrite(h,feat.high_change[bar_idx-3]); idx++;
FileWrite(h,feat.high_change[bar_idx-4]); idx++;
FileWrite(h,feat.mid_point[bar_idx]); idx++;
FileWrite(h,feat.mid_point[bar_idx-1]); idx++;
FileWrite(h,feat.mid_point[bar_idx-2]); idx++;
FileWrite(h,feat.mid_point[bar_idx-3]); idx++;
FileWrite(h,feat.mid_point[bar_idx-4]); idx++;
FileWrite(h,feat.mid_point_prev_2[bar_idx]); idx++;
FileWrite(h,feat.mid_point_prev_2[bar_idx-1]); idx++;
FileWrite(h,feat.mid_point_prev_2[bar_idx-2]); idx++;
FileWrite(h,feat.mid_point_prev_2[bar_idx-3]); idx++;
FileWrite(h,feat.mid_point_prev_2[bar_idx-4]); idx++;
FileWrite(h,feat.body_size_prev[bar_idx]); idx++;
FileWrite(h,feat.body_size_prev[bar_idx-1]); idx++;
FileWrite(h,feat.body_size_prev[bar_idx-2]); idx++;
FileWrite(h,feat.body_size_prev[bar_idx-3]); idx++;
FileWrite(h,feat.body_size_prev[bar_idx-4]); idx++;
FileWrite(h,feat.direction_prev[bar_idx]); idx++;
FileWrite(h,feat.direction_prev[bar_idx-1]); idx++;
FileWrite(h,feat.direction_prev[bar_idx-2]); idx++;
FileWrite(h,feat.direction_prev[bar_idx-3]); idx++;
FileWrite(h,feat.direction_prev[bar_idx-4]); idx++;
FileWrite(h,feat.direction_prev_2[bar_idx]); idx++;
FileWrite(h,feat.direction_prev_2[bar_idx-1]); idx++;
FileWrite(h,feat.direction_prev_2[bar_idx-2]); idx++;
FileWrite(h,feat.direction_prev_2[bar_idx-3]); idx++;
FileWrite(h,feat.direction_prev_2[bar_idx-4]); idx++;
FileWrite(h,feat.body_perc_prev[bar_idx]); idx++;
FileWrite(h,feat.body_perc_prev[bar_idx-1]); idx++;
FileWrite(h,feat.body_perc_prev[bar_idx-2]); idx++;
FileWrite(h,feat.body_perc_prev[bar_idx-3]); idx++;
FileWrite(h,feat.body_perc_prev[bar_idx-4]); idx++;
FileWrite(h,feat.body_perc_prev_2[bar_idx]); idx++;
FileWrite(h,feat.body_perc_prev_2[bar_idx-1]); idx++;
FileWrite(h,feat.body_perc_prev_2[bar_idx-2]); idx++;
FileWrite(h,feat.body_perc_prev_2[bar_idx-3]); idx++;
FileWrite(h,feat.body_perc_prev_2[bar_idx-4]); idx++;
FileWrite(h,feat.HANGING_MAN[bar_idx]); idx++;
FileWrite(h,feat.HANGING_MAN[bar_idx-1]); idx++;
FileWrite(h,feat.HANGING_MAN[bar_idx-2]); idx++;
FileWrite(h,feat.HANGING_MAN[bar_idx-3]); idx++;
FileWrite(h,feat.HANGING_MAN[bar_idx-4]); idx++;
FileWrite(h,feat.SHOOTING_STAR[bar_idx]); idx++;
FileWrite(h,feat.SHOOTING_STAR[bar_idx-1]); idx++;
FileWrite(h,feat.SHOOTING_STAR[bar_idx-2]); idx++;
FileWrite(h,feat.SHOOTING_STAR[bar_idx-3]); idx++;
FileWrite(h,feat.SHOOTING_STAR[bar_idx-4]); idx++;
FileWrite(h,feat.SPINNING_TOP[bar_idx]); idx++;
FileWrite(h,feat.SPINNING_TOP[bar_idx-1]); idx++;
FileWrite(h,feat.SPINNING_TOP[bar_idx-2]); idx++;
FileWrite(h,feat.SPINNING_TOP[bar_idx-3]); idx++;
FileWrite(h,feat.SPINNING_TOP[bar_idx-4]); idx++;
FileWrite(h,feat.MARUBOZU[bar_idx]); idx++;
FileWrite(h,feat.MARUBOZU[bar_idx-1]); idx++;
FileWrite(h,feat.MARUBOZU[bar_idx-2]); idx++;
FileWrite(h,feat.MARUBOZU[bar_idx-3]); idx++;
FileWrite(h,feat.MARUBOZU[bar_idx-4]); idx++;
FileWrite(h,feat.ENGULFING[bar_idx]); idx++;
FileWrite(h,feat.ENGULFING[bar_idx-1]); idx++;
FileWrite(h,feat.ENGULFING[bar_idx-2]); idx++;
FileWrite(h,feat.ENGULFING[bar_idx-3]); idx++;
FileWrite(h,feat.ENGULFING[bar_idx-4]); idx++;
FileWrite(h,feat.TWEEZER_TOP[bar_idx]); idx++;
FileWrite(h,feat.TWEEZER_TOP[bar_idx-1]); idx++;
FileWrite(h,feat.TWEEZER_TOP[bar_idx-2]); idx++;
FileWrite(h,feat.TWEEZER_TOP[bar_idx-3]); idx++;
FileWrite(h,feat.TWEEZER_TOP[bar_idx-4]); idx++;
FileWrite(h,feat.TWEEZER_BOTTOM[bar_idx]); idx++;
FileWrite(h,feat.TWEEZER_BOTTOM[bar_idx-1]); idx++;
FileWrite(h,feat.TWEEZER_BOTTOM[bar_idx-2]); idx++;
FileWrite(h,feat.TWEEZER_BOTTOM[bar_idx-3]); idx++;
FileWrite(h,feat.TWEEZER_BOTTOM[bar_idx-4]); idx++;
FileWrite(h,feat.MORNING_STAR[bar_idx]); idx++;
FileWrite(h,feat.MORNING_STAR[bar_idx-1]); idx++;
FileWrite(h,feat.MORNING_STAR[bar_idx-2]); idx++;
FileWrite(h,feat.MORNING_STAR[bar_idx-3]); idx++;
FileWrite(h,feat.MORNING_STAR[bar_idx-4]); idx++;
FileWrite(h,feat.EVENING_STAR[bar_idx]); idx++;
FileWrite(h,feat.EVENING_STAR[bar_idx-1]); idx++;
FileWrite(h,feat.EVENING_STAR[bar_idx-2]); idx++;
FileWrite(h,feat.EVENING_STAR[bar_idx-3]); idx++;
FileWrite(h,feat.EVENING_STAR[bar_idx-4]); idx++;

FileClose(h);

}






