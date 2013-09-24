/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef ESSENTIA_SCHEDULER_GRAPHUTILS_H
#define ESSENTIA_SCHEDULER_GRAPHUTILS_H

#include <vector>
#include <stack>
#include <string>
#include <algorithm>
#include "network.h"
#include "../streaming/streamingalgorithm.h"

namespace essentia {
namespace scheduler {


template <typename NodeType>
void depthFirstApply(NodeType* root, void (*nodeFunc)(NodeType* node)) {
  if (!root) return;

  std::stack<NodeType*> toVisit;
  std::set<NodeType*> visited;
  toVisit.push(root);

  while (!toVisit.empty()) {
    NodeType* currentNode = toVisit.top();
    toVisit.pop();

    if (visited.find(currentNode) != visited.end()) continue;
    visited.insert(currentNode);

    nodeFunc(currentNode);

    const std::vector<NodeType*>& children = currentNode->children();
    // only add the nodes which have not been previously visited
    // NB: we could let the check on currentNode at the beginning of this loop take care
    //     of this, but doing it here is faster because it spares us adding the node and
    //     removing it immediately after
    for (int i=0; i<(int)children.size(); i++) {
      if (visited.find(children[i]) == visited.end()) {
        toVisit.push(children[i]);
      }
    }
  }
}



template <typename NodeType, typename MappedType>
std::vector<MappedType> depthFirstMap(NodeType* root,
                                      MappedType (*mapFunc)(NodeType* node)) {

  if (!root) return std::vector<MappedType>();

  std::vector<MappedType> result;
  std::stack<NodeType*> toVisit;
  std::set<NodeType*> visited;
  toVisit.push(root);

  while (!toVisit.empty()) {
    NodeType* currentNode = toVisit.top();
    toVisit.pop();

    if (visited.find(currentNode) != visited.end()) continue;
    visited.insert(currentNode);

    result.push_back(mapFunc(currentNode));

    const std::vector<NodeType*>& children = currentNode->children();
    // only add the nodes which have not been previously visited
    // NB: we could let the check on currentNode at the beginning of this loop take care
    //     of this, but doing it here is faster because it spares us adding the node and
    //     removing it immediately after
    for (int i=0; i<(int)children.size(); i++) {
      if (visited.find(children[i]) == visited.end()) {
        toVisit.push(children[i]);
      }
    }
  }

  return result;
}

template <typename NodeType>
NodeType* returnIdentity(NodeType* node) {
  return node;
}

template <typename NodeType>
std::vector<NodeType*> depthFirstSearch(NodeType* root) {
  return depthFirstMap(root, returnIdentity<NodeType>);
}

inline std::string removeNodeIdFromName(const std::string& name) {
  std::string::size_type idpos = std::min(name.find('<'), name.find('['));
  if (idpos == std::string::npos) return name;
  return name.substr(0, idpos);
}

inline std::pair<NetworkNode*, std::string> getIdentityAndName(NetworkNode* node) {
  return std::make_pair(node, removeNodeIdFromName(node->algorithm()->name()));
}

inline streaming::Algorithm* returnAlgorithm(NetworkNode* node) {
  return node->algorithm();
}

template <typename T, typename U>
class ReversePairCompare {
 public:
  bool operator()(const std::pair<T, U>& p1, const std::pair<T, U>& p2) const {
    if (p1.second < p2.second) return true;
    if (p1.second > p2.second) return false;
    return p1.first < p2.first;
  }
};

template <typename NodeType>
void adjacencyMatrix(const std::vector<std::pair<NodeType*, std::string> >& nodes,
                     std::vector<std::vector<bool> >& adjMatrix) {
  int nnodes = nodes.size();
  adjMatrix = std::vector<std::vector<bool> >(nnodes, std::vector<bool>(nnodes, false));

  for (int i=0; i<nnodes; i++) {
    NodeType* start = nodes[i].first;
    const std::vector<NodeType*>& children = start->children();
    for (int j=0; j<(int)children.size(); j++) {
      for (int k=0; k<nnodes; k++) {
        if (children[j] == nodes[k].first) adjMatrix[i][k] = true;
      }
    }
  }
}

// most likely this overload is not very dangerous
template <typename NodeType>
void printAdjacencyMatrix(const std::vector<std::vector<bool> >& adjMatrix,
                          const std::vector<std::pair<NodeType*, std::string> >& nodes) {
  int size = adjMatrix.size();
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      E_DEBUG_NONL(EGraph, (adjMatrix[i][j] ? " 1" : " 0"));
    }
    E_DEBUG_NONL(EGraph, "    " << nodes[i].second << " ->");
    for (int j=0; j<size; j++) {
        E_DEBUG_NONL(EGraph, (adjMatrix[i][j] ? (" " + nodes[j].second) : ""));
    }
    E_DEBUG(EGraph, "");
  }
}

template <typename NodeType>
bool areNetworkTopologiesEqual(NodeType* n1, NodeType* n2) {
  // this function compares both network and decide whether they have the same topology
  // to do this, we apply the following algorithm (not the most efficient, yeah)
  //  - get list of all nodes (if not same number -> different networks)
  //  - order them by alphabetical order (need to be same order)
  //  - for each duplicate node name, generate permutations for this node
  //  - for each permutation, generate adjacency matrix and compare them:
  //    - if there is 1 permutation for which the matrices are equal, then the networks are equal
  //    - if the matrices are different for all the permutations, then the networks are different

  // get the list of nodes and their names
  std::vector<std::pair<NodeType*, std::string> > nodes1 = depthFirstMap(n1, getIdentityAndName);
  std::vector<std::pair<NodeType*, std::string> > nodes2 = depthFirstMap(n2, getIdentityAndName);

  E_DEBUG(EGraph, "Comparing network topologies:");
  E_DEBUG(EGraph, "  n1.size() = " << nodes1.size() << " - n2.size() = " << nodes2.size());

  if (nodes1.size() != nodes2.size()) return false;
  int nnodes = nodes1.size();

  // sort the nodes by their names and compare them
  std::sort(nodes1.begin(), nodes1.end(), ReversePairCompare<NodeType*, std::string>());
  std::sort(nodes2.begin(), nodes2.end(), ReversePairCompare<NodeType*, std::string>());

  E_DEBUG(EGraph, "  nodes1: " << nodes1);
  E_DEBUG(EGraph, "  nodes2: " << nodes2);

  for (int i=0; i<nnodes; i++) {
    if (nodes1[i].second != nodes2[i].second) return false;
  }

  // now try all the permutations of the nodes with the same name and try to look whether there is
  // one for which the adjacency matrices are equal
  std::vector<std::pair<int, int> > same; // ranges for the nodes which have the same name

  int idx = 0;
  while (idx < (nnodes-1)) {
    if (nodes1[idx].second != nodes1[idx+1].second) { idx++; continue; }
    // we have at least 2 nodes with the same name
    int start_idx = idx;
    const std::string& name = nodes1[idx].second;
    idx++;
    while (idx < nnodes && nodes1[idx].second == name) idx++;
    same.push_back(std::make_pair(start_idx, idx));
  }

  // reset the positions for the permutations to work correctly
  for (int i=0; i<(int)same.size(); i++) {
    std::sort(nodes1.begin() + same[i].first,
              nodes1.begin() + same[i].second);
  }

  std::vector<std::vector<bool> > adjMatrix1;
  std::vector<std::vector<bool> > adjMatrix2;

  // 2 will be fixed, compute its adjacency matrix now
  adjacencyMatrix(nodes2, adjMatrix2);


  E_DEBUG(EGraph, "adj matrix 2:");
  printAdjacencyMatrix(adjMatrix2, nodes2);


  // try each permutation and compare adj. matrices
  while (true) {
    // if adjacency matrices are equal, return true;
    adjacencyMatrix(nodes1, adjMatrix1);

    E_DEBUG(EGraph, "comparing with");
    printAdjacencyMatrix(adjMatrix1, nodes1);

    if (adjMatrix1 == adjMatrix2) return true;

    // otherwise, advance to next permutation
    int permidx = 0;
    bool nextgroup = true;
    while (nextgroup) {
      if (permidx == (int)same.size()) return false; // we exhausted all permutations and didn't find a correspondence
      // we want to permute the next group only if the current group finished a full cycle
      nextgroup = !std::next_permutation(nodes1.begin() + same[permidx].first,
                                         nodes1.begin() + same[permidx].second);
      permidx++;
    }
  }

  assert(false);
}

} // namespace scheduler
} // namespace essentia


#endif // ESSENTIA_SCHEDULER_GRAPHUTILS_H
