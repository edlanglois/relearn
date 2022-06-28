use super::{
    testing, BooleanSpace, FeatureSpace, FiniteSpace, IndexSpace, IntervalSpace, LogElementSpace,
    Space, SubsetOrd,
};
use crate::logging::{Id, LogError, LogValue, StatsLogger};
use std::cmp::Ordering;

/// Mock logger for testing `LogElementSpace`
#[derive(Debug, Default)]
struct MockLogger {
    calls: Vec<MockLogCall>,
}

#[derive(Debug, Clone, PartialEq)]
enum MockLogCall {
    GroupStart,
    Log { id: Id, value: LogValue },
    GroupEnd,
    Flush,
}

impl StatsLogger for MockLogger {
    fn group_start(&mut self) {
        self.calls.push(MockLogCall::GroupStart);
    }
    fn group_log(&mut self, id: Id, value: LogValue) -> Result<(), LogError> {
        self.calls.push(MockLogCall::Log { id, value });
        Ok(())
    }
    fn group_end(&mut self) {
        self.calls.push(MockLogCall::GroupEnd);
    }
    fn flush(&mut self) {
        self.calls.push(MockLogCall::Flush);
    }
}

mod unit {
    use super::*;

    #[derive(
        Debug,
        PartialEq,
        Space,
        SubsetOrd,
        FiniteSpace,
        NonEmptySpace,
        SampleSpace,
        FeatureSpace,
        LogElementSpace,
    )]
    struct UnitSpace;

    mod space {
        use super::*;

        #[test]
        fn contains() {
            let _: &dyn Space<Element = ()> = &UnitSpace;

            let s = UnitSpace;
            assert!(s.contains(&()));
        }

        #[test]
        fn contains_samples() {
            testing::check_contains_samples(&UnitSpace, 10);
        }

        features_tests!(f, UnitSpace, (), []);
        batch_features_tests!(b, UnitSpace, [(), (), ()], [[], [], []]);
    }

    mod subset_ord {
        use super::*;

        #[test]
        fn cmp_equal() {
            assert_eq!(UnitSpace.subset_cmp(&UnitSpace), Some(Ordering::Equal));
        }

        #[test]
        fn not_strict_subset() {
            assert!(!UnitSpace.strict_subset_of(&UnitSpace));
        }
    }

    mod finite_space {
        use super::*;

        #[test]
        fn size() {
            assert_eq!(UnitSpace.size(), 1);
        }

        #[test]
        fn to_index() {
            assert_eq!(UnitSpace.to_index(&()), 0);
        }

        #[test]
        fn from_index_valid() {
            assert_eq!(UnitSpace.from_index(0), Some(()));
        }

        #[test]
        fn from_index_invalid() {
            assert_eq!(UnitSpace.from_index(1), None);
        }

        #[test]
        fn from_to_index_iter_size() {
            testing::check_from_to_index_iter_size(&UnitSpace);
        }

        #[test]
        fn from_to_index_random() {
            testing::check_from_to_index_random(&UnitSpace, 10);
        }
    }

    mod feature_space {
        use super::*;

        #[test]
        fn num_features() {
            assert_eq!(UnitSpace.num_features(), 0);
        }
    }

    mod log_element_space {
        use super::*;

        #[test]
        fn log_element() {
            let mut logger = MockLogger::default();
            UnitSpace.log_element("foo", &(), &mut logger).unwrap();
            assert_eq!(
                logger.calls,
                [MockLogCall::GroupStart, MockLogCall::GroupEnd]
            );
        }
    }
}

mod named {
    use super::*;

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    struct NamedStruct {
        a: bool,
        b: usize,
    }

    impl NamedStruct {
        const fn new(a: bool, b: usize) -> Self {
            Self { a, b }
        }
    }

    #[derive(Debug, PartialEq, ProductSpace, FiniteSpace)]
    #[element(NamedStruct)]
    struct NamedStructSpace {
        a: BooleanSpace,
        b: IndexSpace,
    }

    const fn space() -> NamedStructSpace {
        NamedStructSpace {
            a: BooleanSpace,
            b: IndexSpace::new(3),
        }
    }

    mod space {
        use super::*;

        #[test]
        fn contains() {
            let s = space();
            let _: &dyn Space<Element = NamedStruct> = &s;
            assert!(s.contains(&NamedStruct::new(false, 0)));
            assert!(!s.contains(&NamedStruct::new(false, 10)));
        }

        #[test]
        fn contains_samples() {
            testing::check_contains_samples(&space(), 10);
        }
    }

    mod subset_ord {
        use super::*;

        #[test]
        fn equal() {
            assert_eq!(space().subset_cmp(&space()), Some(Ordering::Equal));
        }

        #[test]
        fn strict_subset() {
            let s1 = NamedStructSpace {
                a: BooleanSpace,
                b: IndexSpace::new(3),
            };
            let s2 = NamedStructSpace {
                a: BooleanSpace,
                b: IndexSpace::new(4),
            };
            assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Less));
        }

        #[test]
        fn strict_superset() {
            let s1 = NamedStructSpace {
                a: BooleanSpace,
                b: IndexSpace::new(3),
            };
            let s2 = NamedStructSpace {
                a: BooleanSpace,
                b: IndexSpace::new(2),
            };
            assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Greater));
        }
    }

    mod finite_space {
        use super::*;

        #[test]
        fn size() {
            assert_eq!(space().size(), 6);
        }

        #[test]
        fn to_index() {
            let s = space();
            assert_eq!(s.to_index(&NamedStruct::new(false, 0)), 0);
            assert_eq!(s.to_index(&NamedStruct::new(true, 0)), 1);
            assert_eq!(s.to_index(&NamedStruct::new(false, 1)), 2);
            assert_eq!(s.to_index(&NamedStruct::new(false, 2)), 4);
        }

        #[test]
        fn from_index_valid() {
            let s = space();
            assert_eq!(s.from_index(1), Some(NamedStruct::new(true, 0)));
            assert_eq!(s.from_index(2), Some(NamedStruct::new(false, 1)));
        }

        #[test]
        fn from_index_invalid() {
            let s = space();
            assert_eq!(s.from_index(6), None);
        }

        #[test]
        fn from_to_index_iter_size() {
            testing::check_from_to_index_iter_size(&space());
        }

        #[test]
        fn from_to_index_random() {
            testing::check_from_to_index_random(&space(), 10);
        }
    }

    mod feature_space {
        use super::*;

        #[test]
        fn num_features() {
            assert_eq!(space().num_features(), 4);
        }

        features_tests!(f, space(), NamedStruct::new(true, 1), [1.0, 0.0, 1.0, 0.0]);
        batch_features_tests!(
            b,
            space(),
            [NamedStruct::new(false, 0), NamedStruct::new(true, 2)],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
        );
    }

    mod log_element_space {
        use super::*;

        #[test]
        fn log_element() {
            let mut logger = MockLogger::default();
            space()
                .log_element("foo", &NamedStruct::new(true, 1), &mut logger)
                .unwrap();
            assert_eq!(
                logger.calls,
                [
                    MockLogCall::GroupStart,
                    MockLogCall::Log {
                        id: ["foo", "a"].into_iter().collect(),
                        value: LogValue::Scalar(1.0)
                    },
                    MockLogCall::Log {
                        id: ["foo", "b"].into_iter().collect(),
                        value: LogValue::Index { value: 1, size: 3 }
                    },
                    MockLogCall::GroupEnd,
                ]
            );
        }
    }
}

mod named_generic {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct NamedGeneric<T> {
        inner: T,
    }

    impl<T> NamedGeneric<T> {
        const fn new(inner: T) -> Self {
            Self { inner }
        }
    }

    #[derive(
        Debug,
        PartialEq,
        Space,
        SubsetOrd,
        FiniteSpace,
        NonEmptySpace,
        SampleSpace,
        FeatureSpace,
        LogElementSpace,
    )]
    #[element(NamedGeneric<T::Element>)]
    struct NamedGenericSpace<T> {
        inner: T,
    }

    impl<T> NamedGenericSpace<T> {
        const fn new(inner: T) -> Self {
            Self { inner }
        }
    }

    const fn space() -> NamedGenericSpace<IndexSpace> {
        NamedGenericSpace::new(IndexSpace::new(3))
    }

    mod space {
        use super::*;

        #[test]
        fn contains() {
            let s = space();
            let _: &dyn Space<Element = NamedGeneric<usize>> = &s;

            assert!(s.contains(&NamedGeneric::new(1)));
            assert!(!s.contains(&NamedGeneric::new(4)));
        }

        #[test]
        fn contains_samples() {
            testing::check_contains_samples(&space(), 10);
        }
    }

    mod subset_ord {
        use super::*;

        #[test]
        fn equal() {
            let s = space();
            assert_eq!(s.subset_cmp(&s), Some(Ordering::Equal));
        }

        #[test]
        fn strict_subset() {
            let s1 = NamedGenericSpace::new(IndexSpace::new(3));
            let s2 = NamedGenericSpace::new(IndexSpace::new(4));
            assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Less));
        }

        #[test]
        fn strict_superset() {
            let s1 = NamedGenericSpace::new(IndexSpace::new(3));
            let s2 = NamedGenericSpace::new(IndexSpace::new(2));
            assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Greater));
        }

        #[test]
        fn inner_incomparable() {
            let s1 = NamedGenericSpace::new(IntervalSpace::new(0.0, 2.0));
            let s2 = NamedGenericSpace::new(IntervalSpace::new(1.0, 3.0));
            assert!(s1.subset_cmp(&s2).is_none());
        }
    }

    mod finite_space {
        use super::*;

        #[test]
        fn size() {
            assert_eq!(space().size(), 3);
        }

        #[test]
        fn to_index() {
            let s = space();
            assert_eq!(s.to_index(&NamedGeneric::new(0)), 0);
            assert_eq!(s.to_index(&NamedGeneric::new(1)), 1);
        }

        #[test]
        fn from_index_valid() {
            let s = space();
            assert_eq!(s.from_index(0), Some(NamedGeneric::new(0)));
            assert_eq!(s.from_index(1), Some(NamedGeneric::new(1)));
        }

        #[test]
        fn from_index_invalid() {
            let s = space();
            assert_eq!(s.from_index(3), None);
        }

        #[test]
        fn from_to_index_iter_size() {
            testing::check_from_to_index_iter_size(&space());
        }

        #[test]
        fn from_to_index_random() {
            testing::check_from_to_index_random(&space(), 10);
        }
    }

    mod feature_space {
        use super::*;

        #[test]
        fn num_features() {
            assert_eq!(space().num_features(), 3);
        }

        features_tests!(f, space(), NamedGeneric::new(1), [0.0, 1.0, 0.0]);
        batch_features_tests!(
            b,
            space(),
            [
                NamedGeneric::new(2),
                NamedGeneric::new(0),
                NamedGeneric::new(1)
            ],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
    }

    mod log_element_space {
        use super::*;

        #[test]
        fn log_element() {
            let mut logger = MockLogger::default();
            space()
                .log_element("foo", &NamedGeneric::new(1), &mut logger)
                .unwrap();
            assert_eq!(
                logger.calls,
                [
                    MockLogCall::GroupStart,
                    MockLogCall::Log {
                        id: ["foo", "inner"].into_iter().collect(),
                        value: LogValue::Index { value: 1, size: 3 }
                    },
                    MockLogCall::GroupEnd,
                ]
            );
        }
    }
}

mod unnamed {
    use super::*;

    #[derive(
        Debug,
        PartialEq,
        Space,
        SubsetOrd,
        FiniteSpace,
        NonEmptySpace,
        SampleSpace,
        FeatureSpace,
        LogElementSpace,
    )]
    struct UnnamedStructSpace(BooleanSpace, IndexSpace);

    const fn space() -> UnnamedStructSpace {
        UnnamedStructSpace(BooleanSpace, IndexSpace::new(3))
    }

    mod space {
        use super::*;

        #[test]
        fn contains() {
            let s = space();

            let _: &dyn Space<Element = (bool, usize)> = &s;
            assert!(s.contains(&(false, 0)));
            assert!(!s.contains(&(false, 10)));
        }

        #[test]
        fn contains_samples() {
            testing::check_contains_samples(&space(), 10);
        }
    }

    mod subset_ord {
        use super::*;

        #[test]
        fn equal() {
            let s = space();
            assert_eq!(s.subset_cmp(&s), Some(Ordering::Equal));
        }

        #[test]
        fn strict_subset() {
            let s1 = UnnamedStructSpace(BooleanSpace, IndexSpace::new(3));
            let s2 = UnnamedStructSpace(BooleanSpace, IndexSpace::new(4));
            assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Less));
        }

        #[test]
        fn strict_superset() {
            let s1 = UnnamedStructSpace(BooleanSpace, IndexSpace::new(3));
            let s2 = UnnamedStructSpace(BooleanSpace, IndexSpace::new(2));
            assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Greater));
        }
    }

    mod finite_space {
        use super::*;

        #[test]
        fn size() {
            assert_eq!(space().size(), 6);
        }

        #[test]
        fn to_index() {
            let s = space();
            assert_eq!(s.to_index(&(false, 0)), 0);
            assert_eq!(s.to_index(&(true, 0)), 1);
            assert_eq!(s.to_index(&(false, 1)), 2);
            assert_eq!(s.to_index(&(false, 2)), 4);
        }

        #[test]
        fn from_index_valid() {
            let s = space();
            assert_eq!(s.from_index(1), Some((true, 0)));
            assert_eq!(s.from_index(2), Some((false, 1)));
        }

        #[test]
        fn from_index_invalid() {
            let s = space();
            assert_eq!(s.from_index(6), None);
        }

        #[test]
        fn from_to_index_iter_size() {
            testing::check_from_to_index_iter_size(&space());
        }

        #[test]
        fn from_to_index_random() {
            testing::check_from_to_index_random(&space(), 10);
        }
    }

    mod feature_space {
        use super::*;

        #[test]
        fn num_features() {
            assert_eq!(space().num_features(), 4);
        }

        features_tests!(f, space(), (true, 1), [1.0, 0.0, 1.0, 0.0]);
        batch_features_tests!(
            b,
            space(),
            [(false, 0), (true, 2)],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
        );
    }

    mod log_element_space {
        use super::*;

        #[test]
        fn log_element() {
            let mut logger = MockLogger::default();
            space().log_element("foo", &(true, 1), &mut logger).unwrap();
            assert_eq!(
                logger.calls,
                [
                    MockLogCall::GroupStart,
                    MockLogCall::Log {
                        id: ["foo", "0"].into_iter().collect(),
                        value: LogValue::Scalar(1.0)
                    },
                    MockLogCall::Log {
                        id: ["foo", "1"].into_iter().collect(),
                        value: LogValue::Index { value: 1, size: 3 }
                    },
                    MockLogCall::GroupEnd,
                ]
            );
        }
    }
}

mod unnamed_generic {
    use super::*;

    #[derive(
        Debug,
        PartialEq,
        Space,
        SubsetOrd,
        FiniteSpace,
        NonEmptySpace,
        SampleSpace,
        FeatureSpace,
        LogElementSpace,
    )]
    struct GenericTriple<T, U>(T, U, U);

    const fn space() -> GenericTriple<IndexSpace, BooleanSpace> {
        GenericTriple(IndexSpace::new(3), BooleanSpace, BooleanSpace)
    }

    mod space {
        use super::*;

        #[test]
        fn contains() {
            let s = GenericTriple(BooleanSpace, IndexSpace::new(2), IndexSpace::new(3));
            let _: &dyn Space<Element = (bool, usize, usize)> = &s;

            assert!(s.contains(&(false, 1, 0)));
            assert!(!s.contains(&(false, 2, 0)));
        }

        #[test]
        fn contains_samples() {
            let s = GenericTriple(BooleanSpace, IndexSpace::new(2), IndexSpace::new(3));
            testing::check_contains_samples(&s, 10);
        }
    }

    mod subset_ord {
        use super::*;

        #[test]
        #[allow(clippy::eq_op)]
        fn same_interval_eq() {
            let s = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            assert_eq!(s, s);
        }

        #[test]
        fn same_interval_cmp_equal() {
            let s = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            assert_eq!(s.subset_cmp(&s), Some(Ordering::Equal));
        }

        #[test]
        fn different_interval_ne() {
            let s1 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            let s2 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(0.0, 1.0),
            );
            assert!(s1 != s2);
        }

        #[test]
        fn subset_interval_strict_subset() {
            let s1 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.2, 0.8),
                IntervalSpace::new(2.2, 2.8),
            );
            let s2 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            assert!(s1.strict_subset_of(&s2));
        }

        #[test]
        fn partial_subset_interval_strict_subset() {
            let s1 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.2, 0.8),
                IntervalSpace::new(2.0, 3.0),
            );
            let s2 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            assert!(s1.strict_subset_of(&s2));
        }

        #[test]
        fn superset_interval_strict_superset() {
            let s1 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(-0.2, 1.2),
                IntervalSpace::new(1.8, 3.2),
            );
            let s2 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            assert!(s1.strict_superset_of(&s2));
        }

        #[test]
        fn mixed_subset_superset_incomparable() {
            let s1 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.2, 0.8),
                IntervalSpace::new(1.8, 3.2),
            );
            let s2 = GenericTriple(
                BooleanSpace,
                IntervalSpace::new(0.0, 1.0),
                IntervalSpace::new(2.0, 3.0),
            );
            assert!(s1.subset_cmp(&s2).is_none());
        }
    }

    mod finite_space {
        use super::*;

        #[test]
        fn size() {
            assert_eq!(space().size(), 12);
        }

        #[test]
        fn to_index() {
            let s = space();
            assert_eq!(s.to_index(&(0, false, false)), 0);
            assert_eq!(s.to_index(&(1, false, false)), 1);
            assert_eq!(s.to_index(&(2, false, false)), 2);
            assert_eq!(s.to_index(&(0, true, false)), 3);
            assert_eq!(s.to_index(&(0, false, true)), 6);
            assert_eq!(s.to_index(&(1, false, true)), 7);
        }

        #[test]
        fn from_index_valid() {
            let s = space();
            assert_eq!(s.from_index(0), Some((0, false, false)));
            assert_eq!(s.from_index(4), Some((1, true, false)));
            assert_eq!(s.from_index(11), Some((2, true, true)));
            assert_eq!(s.from_index(12), None);
        }

        #[test]
        fn from_index_invalid() {
            let s = space();
            assert_eq!(s.from_index(12), None);
        }

        #[test]
        fn from_to_index_iter_size() {
            testing::check_from_to_index_iter_size(&space());
        }

        #[test]
        fn from_to_index_random() {
            testing::check_from_to_index_random(&space(), 10);
        }
    }

    mod feature_space {
        use super::*;

        const fn space() -> GenericTriple<IndexSpace, BooleanSpace> {
            GenericTriple(IndexSpace::new(3), BooleanSpace, BooleanSpace)
        }

        #[test]
        fn num_features() {
            assert_eq!(space().num_features(), 5);
        }

        features_tests!(f, space(), (1, true, false), [0.0, 1.0, 0.0, 1.0, 0.0]);
        batch_features_tests!(
            b,
            space(),
            [(1, true, false), (0, false, false)],
            [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
        );
    }

    mod log_element_space {
        use super::*;

        #[test]
        fn log_element() {
            let mut logger = MockLogger::default();
            space()
                .log_element("foo", &(1, true, false), &mut logger)
                .unwrap();
            assert_eq!(
                logger.calls,
                [
                    MockLogCall::GroupStart,
                    MockLogCall::Log {
                        id: ["foo", "0"].into_iter().collect(),
                        value: LogValue::Index { value: 1, size: 3 }
                    },
                    MockLogCall::Log {
                        id: ["foo", "1"].into_iter().collect(),
                        value: LogValue::Scalar(1.0)
                    },
                    MockLogCall::Log {
                        id: ["foo", "2"].into_iter().collect(),
                        value: LogValue::Scalar(0.0)
                    },
                    MockLogCall::GroupEnd,
                ]
            );
        }
    }
}

/// No runtime tests, just make sure everything compiles
///
/// In particular, make sure that size-one tuples are being interpreted as tuples `(x,)` not
/// raw inner values `x`.
mod unnamed_one {
    #[derive(
        Debug, PartialEq, Space, SubsetOrd, FiniteSpace, SampleSpace, FeatureSpace, LogElementSpace,
    )]
    struct UnnamedOne<T>(T);
}
