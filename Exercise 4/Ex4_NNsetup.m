%Template for setting up and training a pattern classification neural net
%which:
%1) performs no normalization or other preprocessing of the input
%2) uses predefined test/va/train sets according to indices
%
%-- A. Dimopoulos, June 2022

%all Data and labels. -->DEFINE THESE<--
Dall;
Tall;

%set indices. -->DEFINE THESE<--
%training indices
TRAIN_IND = TRN_STRT:TRN_END;
%validation indices
VAL_IND = VAL_STRT:VAL_END;
%testing indices
TEST_IND = TEST_STRT:TEST_END;

%declare a classification network with default number of hidden units
n = patternnet;
%ensure that no data normalization or other preprocessing is performed
n.inputs{1}.processFcns = {};
%assign train/val/test sets by index
n.divideFcn = 'divideind';
n.divideParam.trainInd = TRAIN_IND;
n.divideParam.valInd = VAL_IND;
n.divideParam.testInd = TEST_IND;
%configure input and output sizes
n = configure(n,Dall,Tall);
%initialize the network weights
n = init(n);
%train the network
[n,train_record] = train(n,Dall,Tall);