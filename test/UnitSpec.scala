import akka.actor.ActorSystem
import org.scalatestplus.play._
import play.api.test.Helpers._
import play.api.test.FakeRequest

import controllers.PgController
import com.scalaml.nn._

/**
 * Unit tests can run without a full Play application.
 */
class UnitSpec extends PlaySpec {

  "PgController" should {

    "return a valid error with action init" in {
      val pg = new Playground
      val controller = new PgController(stubControllerComponents(), pg)
      val result = controller.test(FakeRequest()) // return JSON string
      (pg.report.lossTest.toDouble < 1) mustBe true
      (pg.report.lossTest.toDouble > 0) mustBe true
    }
  }

  "Playground" should {
  }

}
