;; PDDL Domain for Panda Robot Pick-and-Place
;; Using STRIPS-only (no typed PDDL) for full pyperplan 2.1 compatibility

(define (domain pick-and-place)
  (:requirements :strips)

  ;; Predicates — facts about the world
  (:predicates
    (on ?o ?l)           ;; object ?o is at location ?l
    (holding ?o)          ;; robot is holding object ?o
    (gripper-empty)       ;; gripper is empty
    (at-goal ?o)          ;; object has reached the target
  )

  ;; pick: pick up an object from a location
  (:action pick
    :parameters (?o ?l)
    :precondition (and (on ?o ?l) (gripper-empty))
    :effect       (and (holding ?o)
                       (not (on ?o ?l))
                       (not (gripper-empty)))
  )

  ;; place: set down a held object at a location
  (:action place
    :parameters (?o ?l)
    :precondition (holding ?o)
    :effect       (and (on ?o ?l)
                       (at-goal ?o)
                       (not (holding ?o))
                       (gripper-empty))
  )
)
