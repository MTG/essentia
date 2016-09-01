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

#ifndef ESSENTIA_ALGORITHMFACTORY_H
#define ESSENTIA_ALGORITHMFACTORY_H

#include <map>
#include <sstream>
#include <iostream>
#include "types.h"
#include "essentia.h"
#include "parameter.h"


namespace essentia {

/**
 * Class that also contains static information about the algorithms that
 * shouldn't appear in virtual functions because these should be available
 * without the need to instantiate a concrete Algorithm. (A shame there's
 * no static virtual function in C++)
 * This include: the creator function, the name of the algorithm,
 * a short description of an algorithm and the parameters it needs.
 */
template <typename BaseAlgorithm>
class ESSENTIA_API AlgorithmInfo {
 public:
  typedef BaseAlgorithm* (*AlgorithmCreator)();

  AlgorithmCreator create;
  std::string name; // do we need this one or is it redundant
  std::string description;
  std::string category;
};


/**
 * This factory creates instances of the common BaseAlgorithm interface, while
 * getting information from the ReferenceAlgorithm implementation.
 */
template <typename BaseAlgorithm>
class ESSENTIA_API EssentiaFactory {

  static EssentiaFactory<BaseAlgorithm>* _instance;

 public:

  static void init() {
    if (!_instance) {
      _instance = new EssentiaFactory<BaseAlgorithm>();
    }
  }

  static void shutdown() {
    delete _instance;
    _instance = 0;
  }

  /**
   * Creates an instance of the algorithm specified by its name.
   * All the other overloads of this method do the same thing, and additionally
   * configure the algorithm using the given parameters.
   *
   * @throw EssentiaException in case the algorithm could not be created.
   *        This can happen because the given name is not a valid name, or
   *        if any of the given parameters is not a valid one (ie: it is not
   *        supported by the algorithm, or the value it has been given is not
   *        an accepted one).
   */
  static BaseAlgorithm* create(const std::string& id) {
    return instance().create_i(id);
  }

  /**
   * Deletes the specified Algorithm object and frees its memory.
   * @todo make sure this actually works through dynamic libraries' boundaries.
   */
  static void free(BaseAlgorithm* algo) {
    delete algo;
  }

  /**
   * @todo make this return a const ref so we can use "for k in AlgoFactory::keys()".
   * Returns a list of the available algorithms already registered in the
   * factory.
   */
  static std::vector<std::string> keys();

  /**
   * Returns the AlgorithmInfo structure corresponding to the specified
   * algorithm.
   */
  static const AlgorithmInfo<BaseAlgorithm>& getInfo(const std::string& id) { return instance()._map[id]; }

  /**
   * The registrar class that's used to easily register objects in the factory.
   */
  template <typename ConcreteProduct, typename ReferenceConcreteProduct = ConcreteProduct>
  class Registrar {

   public:
    Registrar() {
      // create the object to be inserted into the factory
      // with all the necessary information
      AlgorithmInfo<BaseAlgorithm> entry;
      entry.create = &create;
      entry.name = ReferenceConcreteProduct::name;
      entry.description = ReferenceConcreteProduct::description;
      entry.category = ReferenceConcreteProduct::category;

      // insert object into the factory, or overwrite the existing one if any
      CreatorMap& algoMap = EssentiaFactory::instance()._map;
      if (algoMap.find(entry.name) != algoMap.end()) {
        E_WARNING("Overwriting registered algorithm " << entry.name);
        algoMap[entry.name] = entry;
      }
      else {
        algoMap.insert(entry.name, entry);
        E_DEBUG(EFactory, "Registered algorithm " << entry.name);
      }
    }

    static BaseAlgorithm* create() {
      return new ConcreteProduct;
    }
  };


  static EssentiaFactory& instance();

 protected:
  // protected constructor to ensure singleton.
  EssentiaFactory() {}
  EssentiaFactory(EssentiaFactory&);

  BaseAlgorithm* create_i(const std::string& id) const;

  typedef EssentiaMap<std::string, AlgorithmInfo<BaseAlgorithm>, string_cmp> CreatorMap;
  CreatorMap _map;



  // conveniency functions that allow to configure an algorithm directly at
  // creation time
#define CREATE static BaseAlgorithm* create(const std::string& id
#define CBEG ) { return instance().create_i(id
#define P(n) , const std::string& name##n, const Parameter& value##n
#define AP(n) , name##n, value##n
#define CEND ); }

 public:

  CREATE P(1) CBEG AP(1) CEND
  CREATE P(1) P(2) CBEG AP(1) AP(2) CEND
  CREATE P(1) P(2) P(3) CBEG AP(1) AP(2) AP(3) CEND
  CREATE P(1) P(2) P(3) P(4) CBEG AP(1) AP(2) AP(3) AP(4) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) CBEG AP(1) AP(2) AP(3) AP(4) AP(5) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) AP(15) CEND
  CREATE P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15) P(16)
    CBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) AP(15) AP(16) CEND


#define CREATEI BaseAlgorithm* create_i(const std::string& id
#define CENDI ) const;

 protected:

  CREATEI P(1) CENDI
  CREATEI P(1) P(2) CENDI
  CREATEI P(1) P(2) P(3) CENDI
  CREATEI P(1) P(2) P(3) P(4) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15) CENDI
  CREATEI P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15) P(16) CENDI

#undef CENDI
#undef CREATEI
#undef CEND
#undef AP
#undef P
#undef CBEG
#undef CREATE

};

} // namespace essentia


// include these here because most likely a user of the AlgorithmFactory would want to use the
// returned algorithm :)
#include "algorithm.h"
#include "streaming/streamingalgorithm.h"

namespace essentia {

namespace standard {
  typedef EssentiaFactory<Algorithm> AlgorithmFactory;
}

namespace streaming {
  typedef EssentiaFactory<Algorithm> AlgorithmFactory;
}

} // namespace essentia


// include implementation, because the factory is now templated
#include "algorithmfactory_impl.h"

#endif // ESSENTIA_ALGORITHMFACTORY_H
