/*
 * scoreTree.h
 *
 *  Created on: Aug 16, 2015
 *      Author: jahnka
 */

#ifndef SCORETREE_H_
#define SCORETREE_H_

double scoreTree(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector, double bestScore);
double scoreTree2(int n, int m, double** logAltLL, double** logRefLL, char type, int* parentVector, double bestScore, int n2);
double scoreTreeFast(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector);
double scoreTreeFast2(int n, int m, double** logAltLL, double** logRefLL, char type, int* parentVector, int n2);
double maxScoreTreeFast(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
double maxScoreTreeFast2(int n, int m, double** logAltLL, double** logRefLL, int* parent, int* bft, int n2);
double sumScoreTreeFast(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
double* getAttachmentScoresFast(int*parent, int n, double** logScores, int* dataVector, int*bft);
double* getAttachmentScoresFast2(int*parent, int n, double* logAltLLvector, double* logRefLLvector, int*bft, int n2);
double rootAttachementScore(int n, double** logScores, int* mutationVector);
double rootAttachementScore2(int n, double* logRefLLvector);
double scoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, char type, int* parentVector);
double maxScoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
double sumScoreTreeAccurate(int n, int m, double** logScores, int** dataMatrix, int* parent, int* bft);
int** getBestAttachmentScoreAccurate(int** scoreMatrix, int* parent, int n, double** logScores, int* dataVector, int* bft);
int*** getAttachmentMatrices(int* parent, int n, int* dataVector, int* bft);
double getTrueScore(int** matrix, double** logScores);
double getSumAttachmentScoreAccurate(int* parent, int n, double** logScores, int* dataVector, int* bft);
double** getLogScores(double FD, double AD1, double AD2, double CC);
void updateLogScores(double** logScores, double newAD);
double** getScores(double FD, double AD1, double AD2, double CC);
double* getTrueScores(int*** matrix, int n, double** logScores);
void printLogScores(double** logScores);



#endif /* SCORETREE_H_ */

