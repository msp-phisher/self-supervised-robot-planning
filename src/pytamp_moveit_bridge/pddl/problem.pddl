;; PDDL Problem Template for pyperplan (STRIPS)
;; This is a static reference. The planner node generates this at runtime.

(define (problem pick-bolt-symbolic)
  (:domain pick-and-place)

  (:objects bolt_1 table target)

  (:init
    (on bolt_1 table)
    (gripper-empty)
  )

  (:goal
    (at-goal bolt_1)
  )
)
