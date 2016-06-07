/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_ALGORITHMFACTORY_CPP
#define ESSENTIA_ALGORITHMFACTORY_CPP

namespace essentia {

template <typename BaseAlgorithm>
EssentiaFactory<BaseAlgorithm>& EssentiaFactory<BaseAlgorithm>::instance() {
  if (!_instance) {
    throw EssentiaException("You haven't initialized the factory yet... Please do it now!");
  }
  return *_instance;
}

template <typename BaseAlgorithm>
std::vector<std::string> EssentiaFactory<BaseAlgorithm>::keys() {
  std::vector<std::string> result;
  const CreatorMap& m = instance()._map;
  for (typename CreatorMap::const_iterator it = m.begin(); it != m.end(); ++it) {
    result.push_back(it->first);
  }
  return result;
}

template <typename BaseAlgorithm>
BaseAlgorithm* EssentiaFactory<BaseAlgorithm>::create_i(const std::string& id) const {
  E_DEBUG(EFactory, BaseAlgorithm::processingMode << ": Creating algorithm: " << id);

  typename CreatorMap::const_iterator it = _map.find(id);
  if (it == _map.end()) {
    std::ostringstream msg;
    msg << "Identifier '" << id << "' not found in registry...\n";
    msg << "Available algorithms:";
    for (it=_map.begin(); it!=_map.end(); ++it) {
      msg << ' ' << it->first;
    }
    throw EssentiaException(msg);
  }

  E_DEBUG_INDENT;
  BaseAlgorithm* algo = it->second.create();
  E_DEBUG_OUTDENT;

  // adds the name of the algorithm to itself so it knows it.
  algo->setName(id);

  // declare the acceptable parameters for this algorithm. It would be nicer to
  // have this automatically done in the Configurable constructor, but we cannot
  // call  abstract virtual functions from the base constructor.
  algo->declareParameters();

  // configure with default parameters to ensure we're not in an undefined state.
  // This should never throw an exception. If it does, explain why it should
  // absolutely be fixed.
  try {
    // default parameters should have been filled by the call to declareParameters
    // from the constructor, so there is no need to make a copy of them, just call
    // arg-less version of configure()
    E_DEBUG(EFactory, BaseAlgorithm::processingMode << ": Configuring " << id << " with default parameters");
    algo->configure();
  }
  catch (EssentiaException& e) {
    // We should never arrive here, because it means that we can have algorithms
    // which are not configured at all, hence in an invalid state. This cannot
    // happen, hence the message explaining why and we rethrow the exception.
    std::ostringstream msg;
    msg << "ERROR: Algorithm " << id << " could not be configured using default values.\n"
        << "       If it doesn't make sense for an algorithm to be configured with\n"
        << "       default values, then it should still be able to be instantiated, and\n"
        << "       it is your responsibility to keep track of the fact that it should\n"
        << "       currently be impossible to call it (for example, by checking if the state\n"
        << "       is valid upon entering the process() method).\n\n"
        << e.what();
    throw EssentiaException(msg);
  }

  E_DEBUG(EFactory, BaseAlgorithm::processingMode << ": Creating " << id << " ok!");

  return algo;
}


#define CREATE_I template <typename BaseAlgorithm> BaseAlgorithm* EssentiaFactory<BaseAlgorithm>::create_i(const std::string& id
#define P(n) , const std::string& name##n, const Parameter& value##n
#define AP(n) params.add(name##n, value##n);

#define CREATE_I_BEG ) const {                                                                              \
  E_DEBUG(EFactory, BaseAlgorithm::processingMode << ": Creating algorithm: " << id);                       \
  typename CreatorMap::const_iterator it = _map.find(id);                                                   \
  if (it == _map.end()) {                                                                                   \
    std::ostringstream msg;                                                                                 \
    msg << "Identifier '" << id << "' not found in registry...\n";                                          \
    msg << "Available algorithms:";                                                                         \
    for (it=_map.begin(); it!=_map.end(); ++it) {                                                           \
      msg << ' ' << it->first;                                                                              \
    }                                                                                                       \
    throw EssentiaException(msg);                                                                           \
  }                                                                                                         \
  E_DEBUG_INDENT;                                                                                           \
  BaseAlgorithm* algo = it->second.create();                                                                \
  E_DEBUG_OUTDENT;                                                                                          \
  algo->setName(id);                                                                                        \
  algo->declareParameters();                                                                                \
  ParameterMap params;

#define CREATE_I_END                                                                                        \
  algo->setParameters(params);                                                                              \
  E_DEBUG(EFactory, BaseAlgorithm::processingMode << ": Configuring " << id << " with default parameters"); \
  algo->configure();                                                                                        \
  E_DEBUG(EFactory, BaseAlgorithm::processingMode << ": Creating " << id << " ok!");                        \
  return algo;                                                                                              \
}


CREATE_I P(1) CREATE_I_BEG AP(1) CREATE_I_END
CREATE_I P(1) P(2) CREATE_I_BEG AP(1) AP(2) CREATE_I_END
CREATE_I P(1) P(2) P(3) CREATE_I_BEG AP(1) AP(2) AP(3) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) AP(15) CREATE_I_END
CREATE_I P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15) P(16)
  CREATE_I_BEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) AP(15) AP(16) CREATE_I_END


#undef CREATE_I_END
#undef CREATE_I_BODY
#undef AP
#undef P
#undef CREATE_I

} // namespace essentia

#endif // ESSENTIA_ALGORITHMFACTORY_CPP
