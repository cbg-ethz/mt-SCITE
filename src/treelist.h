/*
 * treelist.h
 *
 *  Created on: Mar 27, 2015
 *      Author: jahnka
 */

#include <string>
//#include <iostream>
//#include <sstream>

#ifndef TREELIST_H
#define TREELIST_H

struct treeBeta
{
	int* tree;
    double beta;
};

struct Tree
{
    int* tree;
};

void updateTreeList(std::vector<struct treeBeta>& bestTrees, int* currTreeParentVec, int n, double currScore, double bestScore, double beta);
void updateTreeList2(std::vector<struct Tree>& bestTrees, int* currTreeParentVec, int n, double currScore, double bestScore);
void resetTreeList(std::vector<struct treeBeta>& bestTrees, int* newBestTree, int n, double beta);
void resetTreeList2(std::vector<struct Tree>& bestTrees, int* newBestTree, int n);
void emptyVectorFast(std::vector<struct treeBeta>& optimalTrees, int n);
void emptyVectorFast2(std::vector<struct Tree>& optimalTrees, int n);
void emptyTreeList(std::vector<int*>& optimalTrees, int n);
struct treeBeta createNewTreeListElement(int* tree, int n, double beta);
struct Tree createNewTreeListElement2(int* tree, int n);
bool isDuplicateTreeFast(std::vector<struct treeBeta> &optimalTrees, int* newTree, int n);
bool isDuplicateTreeFast2(std::vector<struct Tree> &optimalTrees, int* newTree, int n);

#endif
